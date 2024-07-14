import io
import logging
import math
import os
import re
import xml.dom
import xml.dom.minidom
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from enum import Enum
from glob import glob
from operator import itemgetter
from shutil import copy2
from typing import Optional

import ijson
import ujson as json
from PIL import Image
from label_studio_sdk.converter import brush
from label_studio_sdk.converter.audio import convert_to_asr_json_manifest
from label_studio_sdk.converter.exports import csv2
from label_studio_sdk.converter.utils import (
    parse_config,
    create_tokens_and_tags,
    download,
    get_image_size_and_channels,
    ensure_dir,
    get_polygon_area,
    get_polygon_bounding_box,
    get_annotator,
    get_json_root_type,
    prettify_result,
    convert_annotation_to_yolo,
    convert_annotation_to_yolo_obb,
)

logger = logging.getLogger(__name__)


class FormatNotSupportedError(NotImplementedError):
    pass


class Format(Enum):
    YOLO = 1
    

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return Format[s]
        except KeyError:
            raise ValueError()


class Converter(object):
    _FORMAT_INFO = {
        Format.YOLO: {
            "title": "YOLO",
            "description": "Popular TXT format is created for each image file. Each txt file contains annotations for "
            "the corresponding image file, that is object class, object coordinates, height & width.",
            "tags": ["image segmentation", "object detection"],
        }
    }

    def all_formats(self):
        return self._FORMAT_INFO

    def __init__(
        self,
        config,
        project_dir,
        output_tags=None,
        upload_dir=None,
        download_resources=True,
    ):
        """Initialize Label Studio Converter for Exports

        :param config: string or dict: XML string with Label studio labeling config or path to this file or parsed_config
        :param project_dir: upload root directory for images, audio and other labeling files
        :param output_tags: it will be calculated automatically, contains label names
        :param upload_dir: upload root directory with files that were imported using LS GUI
        :param download_resources: if True, LS will try to download images, audio, etc and include them to export
        """
        self.project_dir = project_dir
        self.upload_dir = upload_dir
        self.download_resources = download_resources
        self._schema = None

        if isinstance(config, dict):
            self._schema = config
        elif isinstance(config, str):
            if os.path.isfile(config):
                with io.open(config) as f:
                    config_string = f.read()
            else:
                config_string = config
            self._schema = parse_config(config_string)

        if self._schema is None:
            logger.warning(
                "Label config or schema for Converter is not provided, "
                "it might be critical for some export formats, now set schema to empty dict"
            )
            self._schema = {}

        self._data_keys, self._output_tags = self._get_data_keys_and_output_tags(
            output_tags
        )
        self._supported_formats = self._get_supported_formats()

    def convert(self, input_data, output_data, format, is_dir=True, **kwargs):
        if isinstance(format, str):
            format = Format.from_string(format)

        if format == Format.YOLO:
            image_dir = kwargs.get("image_dir")
            label_dir = kwargs.get("label_dir")
            self.convert_to_yolo(
                input_data,
                output_data,
                output_image_dir=image_dir,
                output_label_dir=label_dir,
                is_dir=is_dir
            )
        

    def _get_data_keys_and_output_tags(self, output_tags=None):
        data_keys = set()
        output_tag_names = []
        if output_tags is not None:
            for tag in output_tags:
                if tag not in self._schema:
                    logger.debug(
                        'Specified tag "{tag}" not found in config schema: '
                        "available options are {schema_keys}".format(
                            tag=tag, schema_keys=str(list(self._schema.keys()))
                        )
                    )
        for name, info in self._schema.items():
            if output_tags is not None and name not in output_tags:
                continue
            data_keys |= set(map(itemgetter("value"), info["inputs"]))
            output_tag_names.append(name)

        return list(data_keys), output_tag_names

    def _get_supported_formats(self):
       
        output_tag_types = set()
        input_tag_types = set()
        for info in self._schema.values():
            output_tag_types.add(info["type"])
            for input_tag in info["inputs"]:
                if input_tag["type"] == "Text" and input_tag.get("valueType") == "url":
                    logger.error('valueType="url" are not supported for text inputs')
                    continue
                input_tag_types.add(input_tag["type"])

        all_formats = [f.name for f in Format]
        
        if not (
            "Image" in input_tag_types
            and (
                "RectangleLabels" in output_tag_types
                or "PolygonLabels" in output_tag_types
            )
            or "Rectangle" in output_tag_types
            and "Labels" in output_tag_types
            or "PolygonLabels" in output_tag_types
            and "Labels" in output_tag_types
        ):
            all_formats.remove(Format.YOLO.name)
        

        return all_formats

    @property
    def supported_formats(self):
        return self._supported_formats

    def iter_from_dir(self, input_dir):
        if not os.path.exists(input_dir):
            raise FileNotFoundError(
                "{input_dir} doesn't exist".format(input_dir=input_dir)
            )
        for json_file in glob(os.path.join(input_dir, "*.json")):
            for item in self.iter_from_json_file(json_file):
                if item:
                    yield item

    def iter_from_json_file(self, json_file):
        """Extract annotation results from json file

        param json_file: path to task list or dict with annotations
        """
        data_type = get_json_root_type(json_file)

        # one task
        if data_type == "dict":
            with open(json_file, "r") as json_file:
                data = json.load(json_file)
            for item in self.annotation_result_from_task(data):
                yield item

        # many tasks
        elif data_type == "list":
            with io.open(json_file, "rb") as f:
                logger.debug(f"ijson backend in use: {ijson.backend}")
                data = ijson.items(
                    f, "item", use_float=True
                )  # 'item' means to read array of dicts
                for task in data:
                    for item in self.annotation_result_from_task(task):
                        if item is not None:
                            yield item

    def _maybe_matching_tag_from_schema(self, from_name: str) -> Optional[str]:
        """If the from name exactly matches an output tag from the schema, return that tag.

        Otherwise, certain tags (like those from Repeater) contain
        placeholders like {{idx}}. Such placeholders are mapped to a regex in self._schema.
        For example, if "my_output_tag_{{idx}}" is a tag in the schema,
        then the from_name "my_output_tag_0" should match it, and we should return "my_output_tag_{{idx}}".
        """

        for tag_name, tag_info in self._schema.items():
            if tag_name == from_name:
                return tag_name

            if not tag_info.get("regex"):
                continue

            tag_name_pattern = tag_name
            for variable, regex in tag_info["regex"].items():
                tag_name_pattern = tag_name_pattern.replace(variable, regex)

            if re.compile(tag_name_pattern).match(from_name):
                return tag_name

        return None

    def annotation_result_from_task(self, task):
        has_annotations = "completions" in task or "annotations" in task
        if not has_annotations:
            logger.warning(
                'Each task dict item should contain "annotations" or "completions" [deprecated], '
                "where value is list of dicts"
            )
            return None

        # get last not skipped completion and make result from it
        annotations = (
            task["annotations"] if "annotations" in task else task["completions"]
        )

        # return task with empty annotations
        if not annotations:
            data = Converter.get_data(task, {}, {})
            yield data

        # skip cancelled annotations
        cancelled = lambda x: not (
            x.get("skipped", False) or x.get("was_cancelled", False)
        )
        annotations = list(filter(cancelled, annotations))
        if not annotations:
            return None

        # sort by creation time
        annotations = sorted(
            annotations, key=lambda x: x.get("created_at", 0), reverse=True
        )

        for annotation in annotations:
            result = annotation["result"]
            outputs = defaultdict(list)

            # get results only as output
            for r in result:
                if "from_name" in r and (
                    tag_name := self._maybe_matching_tag_from_schema(r["from_name"])
                ):
                    v = deepcopy(r["value"])
                    v["type"] = self._schema[tag_name]["type"]
                    if "original_width" in r:
                        v["original_width"] = r["original_width"]
                    if "original_height" in r:
                        v["original_height"] = r["original_height"]
                    outputs[r["from_name"]].append(v)

            data = Converter.get_data(task, outputs, annotation)
            if "agreement" in task:
                data["agreement"] = task["agreement"]
            yield data

    @staticmethod
    def get_data(task, outputs, annotation):
        return {
            "id": task["id"],
            "input": task["data"],
            "output": outputs or {},
            "completed_by": annotation.get("completed_by", {}),
            "annotation_id": annotation.get("id"),
            "created_at": annotation.get("created_at"),
            "updated_at": annotation.get("updated_at"),
            "lead_time": annotation.get("lead_time"),
            "history": annotation.get("history"),
            "was_cancelled": annotation.get("was_cancelled"),
        }

    def _check_format(self, fmt):
        pass


    def convert_to_yolo(
            self,
            input_data,
            output_dir,
            output_image_dir=None,
            output_label_dir=None,
            is_dir=True,
            split_labelers=False,
        ):
            """Convert data in a specific format to the YOLO format.

            Parameters
            ----------
            input_data : str
                The input data, either a directory or a JSON file.
            output_dir : str
                The directory to store the output files in.
            output_image_dir : str, optional
                The directory to store the image files in. If not provided, it will default to a subdirectory called 'images' in output_dir.
            output_label_dir : str, optional
                The directory to store the label files in. If not provided, it will default to a subdirectory called 'labels' in output_dir.
            is_dir : bool, optional
                A boolean indicating whether `input_data` is a directory (True) or a JSON file (False).
            split_labelers : bool, optional
                A boolean indicating whether to create a dedicated subfolder for each labeler in the output label directory.
            """
            self._check_format(Format.YOLO)
            ensure_dir(output_dir)
            data_yaml_file = os.path.join(output_dir, 'data.yaml')
            if output_image_dir is not None:
                ensure_dir(output_image_dir)
            else:
                output_image_dir = os.path.join(output_dir, 'images')
                os.makedirs(output_image_dir, exist_ok=True)
            if output_label_dir is not None:
                ensure_dir(output_label_dir)
            else:
                output_label_dir = os.path.join(output_dir, 'labels')
                os.makedirs(output_label_dir, exist_ok=True)
            categories, category_name_to_id = self._get_labels()
            data_key = self._data_keys[0]
            item_iterator = (
                self.iter_from_dir(input_data)
                if is_dir
                else self.iter_from_json_file(input_data)
            )
            for item_idx, item in enumerate(item_iterator):
                # get image path and label file path
                image_path = item['input'][data_key]
                # download image
                if not os.path.exists(image_path):
                    try:
                        image_path = download(
                            image_path,
                            output_image_dir,
                            project_dir=self.project_dir,
                            return_relative_path=True,
                            upload_dir=self.upload_dir,
                            download_resources=self.download_resources,
                        )
                    except:
                        logger.info(
                            'Unable to download {image_path}. The item {item} will be skipped'.format(
                                image_path=image_path, item=item
                            ),
                            exc_info=True,
                        )

                # create dedicated subfolder for each labeler if split_labelers=True
                labeler_subfolder = str(item['completed_by']) if split_labelers else ''
                os.makedirs(
                    os.path.join(output_label_dir, labeler_subfolder), exist_ok=True
                )

                # identify label file path
                filename = os.path.splitext(os.path.basename(image_path))[0]
                filename = filename[
                    0 : 255 - 4
                ]  # urls might be too long, use 255 bytes (-4 for .txt) limit for filenames
                label_path = os.path.join(
                    output_label_dir, labeler_subfolder, filename + '.txt'
                )

                # Skip tasks without annotations
                if not item['output']:
                    logger.warning('No completions found for item #' + str(item_idx))
                    if not os.path.exists(label_path):
                        with open(label_path, 'x'):
                            pass
                    continue

                # concatenate results over all tag names
                labels = []
                for key in item['output']:
                    labels += item['output'][key]

                if len(labels) == 0:
                    logger.warning(f'Empty bboxes for {item["output"]}')
                    if not os.path.exists(label_path):
                        with open(label_path, 'x'):
                            pass
                    continue

                annotations = []
                for label in labels:
                    category_name = None
                    category_names = []  # considering multi-label
                    for key in ['rectanglelabels', 'polygonlabels', 'labels']:
                        if key in label and len(label[key]) > 0:
                            # change to save multi-label
                            for category_name in label[key]:
                                category_names.append(category_name)

                    if len(category_names) == 0:
                        logger.debug(
                            "Unknown label type or labels are empty: " + str(label)
                        )
                        continue

                    for category_name in category_names:
                        if category_name not in category_name_to_id:
                            category_id = len(categories)
                            category_name_to_id[category_name] = category_id
                            categories.append({'id': category_id, 'name': category_name})
                        category_id = category_name_to_id[category_name]

                        if (
                            "rectanglelabels" in label
                            or 'rectangle' in label
                            or 'labels' in label
                        ):
                            annotation = convert_annotation_to_yolo(label)

                            if annotation == None:
                                continue

                            x, y, w, h, = annotation
                            annotations.append([category_id, x, y, w, h])
                                

                        elif "polygonlabels" in label or 'polygon' in label:
                            points_abs = [(x / 100, y / 100) for x, y in label["points"]]
                            annotations.append(
                                [category_id]
                                + [coord for point in points_abs for coord in point]
                            )
                        else:
                            raise ValueError(f"Unknown label type {label}")
                with open(label_path, 'w') as f:
                    for annotation in annotations:
                        for idx, l in enumerate(annotation):
                            if idx == len(annotation) - 1:
                                f.write(f"{l}\n")
                            else:
                                f.write(f"{l} ")
            with open(data_yaml_file, 'w') as f:
                f.write('images: ../images\n')
                f.write('labels: ../labels\n')
                f.write(f'nc: {len(categories)}\n')
                f.write('names: [')
                for i, category in enumerate(categories):
                    if i > 0:
                        f.write(', ')
                    f.write(f"'{category['name']}'")
                f.write(']\n\n')
                f.write('info:\n')
                f.write(f"  year: {datetime.now().year}\n")
                f.write('  version: 1.0\n')

    @staticmethod
    def rotated_rectangle(label):
        if not (
            "x" in label and "y" in label and "width" in label and "height" in label
        ):
            return None

        label_x, label_y, label_w, label_h, label_r = (
            label["x"],
            label["y"],
            label["width"],
            label["height"],
            label["rotation"] if "rotation" in label else 0.0,
        )

        if abs(label_r) > 0:
            alpha = math.atan(label_h / label_w)
            beta = math.pi * (
                label_r / 180
            )  # Label studio defines the angle towards the vertical axis

            radius = math.sqrt((label_w / 2) ** 2 + (label_h / 2) ** 2)

            # Label studio saves the position of top left corner after rotation
            x_0 = (
                label_x
                - radius
                * (math.cos(math.pi - alpha - beta) - math.cos(math.pi - alpha))
                + label_w / 2
            )
            y_0 = (
                label_y
                + radius
                * (math.sin(math.pi - alpha - beta) - math.sin(math.pi - alpha))
                + label_h / 2
            )

            theta_1 = alpha + beta
            theta_2 = math.pi - alpha + beta
            theta_3 = math.pi + alpha + beta
            theta_4 = 2 * math.pi - alpha + beta

            x_coord = [
                x_0 + radius * math.cos(theta_1),
                x_0 + radius * math.cos(theta_2),
                x_0 + radius * math.cos(theta_3),
                x_0 + radius * math.cos(theta_4),
            ]
            y_coord = [
                y_0 + radius * math.sin(theta_1),
                y_0 + radius * math.sin(theta_2),
                y_0 + radius * math.sin(theta_3),
                y_0 + radius * math.sin(theta_4),
            ]

            label_x = min(x_coord)
            label_y = min(y_coord)
            label_w = max(x_coord) - label_x
            label_h = max(y_coord) - label_y

        return label_x, label_y, label_w, label_h

   
    def _get_labels(self):
        labels = set()
        categories = list()
        category_name_to_id = dict()

        for name, info in self._schema.items():
            labels |= set(info["labels"])
            attrs = info["labels_attrs"]
            for label in attrs:
                if attrs[label].get("category"):
                    categories.append(
                        {"id": attrs[label].get("category"), "name": label}
                    )
                    category_name_to_id[label] = attrs[label].get("category")
        labels_to_add = set(labels) - set(list(category_name_to_id.keys()))
        labels_to_add = sorted(list(labels_to_add))
        idx = 0
        while idx in list(category_name_to_id.values()):
            idx += 1
        for label in labels_to_add:
            categories.append({"id": idx, "name": label})
            category_name_to_id[label] = idx
            idx += 1
            while idx in list(category_name_to_id.values()):
                idx += 1
        return categories, category_name_to_id
