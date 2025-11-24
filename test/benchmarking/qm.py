#!/usr/bin/env python3
# Copyright(c) 2025 Meta Platforms, Inc. and affiliates.
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# https://www.aomedia.org/license/patent-license.

import os

import pathlib
import subprocess
from logging import Logger
from typing import Dict, List, Optional, Union

import utils


class QualityMetric:
    def __init__(self, ref_file: str, distorted_file: str):
        self.ref_file = ref_file
        self.distorted_file = distorted_file

    def calculate(self) -> Union[Optional[float], Optional[Dict[str, float]]]:
        raise NotImplementedError("Subclasses must implement calculate()")

    @classmethod
    def get_name(cls) -> str:
        raise NotImplementedError("Subclasses must implement get_name()")


class SSIMULACRA2Metric(QualityMetric):
    def __init__(
        self,
        ref_file: str,
        distorted_file: str,
        logger: Logger,
        ssimulacra2_bin: str,
        artifacts_path: str,
    ):
        super().__init__(ref_file, distorted_file)
        self.ssimulacra2_bin = ssimulacra2_bin
        self.artifacts_path = artifacts_path
        self.logger = logger

    def calculate(self) -> Optional[float]:
        command: List[str] = [self.ssimulacra2_bin, self.ref_file, self.distorted_file]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            self.logger.error(
                f"Error calculating SSIMULACRA2: " f"{result.stderr.strip()}"
            )
            return None

        ref_stem = pathlib.Path(self.ref_file).stem
        out_path = os.path.join(self.artifacts_path, f"{ref_stem}.ssimulcra")
        with open(out_path, "w") as f:
            f.write(result.stdout)

        return float(result.stdout.strip())

    @classmethod
    def get_name(cls) -> str:
        return "SSIMULACRA2"


class VMAFMetric(QualityMetric):
    def __init__(
        self,
        ref_file: str,
        distorted_file: str,
        logger: Logger,
        vmaf_bin: str,
        aom_ctc_model: str = "v6.0",
        artifacts_path: str = "",
    ):
        super().__init__(ref_file=ref_file, distorted_file=distorted_file)
        self.vmaf_bin = vmaf_bin
        self.aom_ctc_model = aom_ctc_model
        self.artifacts_path = artifacts_path
        self.logger = logger

    def calculate(self) -> Optional[Dict[str, float]]:
        ref_stem = pathlib.Path(self.ref_file).stem
        out_path = os.path.join(self.artifacts_path, f"{ref_stem}.xml")

        try:
            command: List[str] = [
                self.vmaf_bin,
                "-r",
                self.ref_file,
                "-d",
                self.distorted_file,
                "-o",
                out_path,
                "--xml",
                "--aom_ctc",
                self.aom_ctc_model,
            ]

            if self.ref_file.endswith(".yuv"):
                width, height, _ = utils.get_file_desc(os.path.basename(self.ref_file))
                command += [
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "--pixel_format",
                    "420",
                    "--bitdepth",
                    "8",
                ]

            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode != 0:
                self.logger.error(f"Error calculating VMAF: {result.stderr.strip()}")
                return None

            import xml.etree.ElementTree as ET

            tree = ET.parse(out_path)
            vmaf_data = tree.getroot()

            metrics = {}

            # Find all metric elements in pooled_metrics section
            for metric_elem in vmaf_data.findall("./pooled_metrics/metric"):
                name = metric_elem.get("name", "")
                mean = float(metric_elem.get("mean", 0))
                metrics[name] = mean
            return metrics

        except Exception as e:
            self.logger.error(f"Error parsing VMAF output: {e}")
            return None

    @classmethod
    def get_name(cls) -> str:
        return "VMAF"


class PSNRMetric(QualityMetric):
    def __init__(
        self,
        ref_file: str,
        distorted_file: str,
        logger: Logger,
        vmaf_bin: str,
        artifacts_path: str = "",
    ):
        super().__init__(ref_file=ref_file, distorted_file=distorted_file)
        self.vmaf_bin = vmaf_bin
        self.artifacts_path = artifacts_path
        self.logger = logger

    def calculate(self) -> Optional[Dict[str, float]]:
        ref_stem = pathlib.Path(self.ref_file).stem
        out_path = os.path.join(self.artifacts_path, f"{ref_stem}.xml")

        try:
            command: List[str] = [
                self.vmaf_bin,
                "-r",
                self.ref_file,
                "-d",
                self.distorted_file,
                "-o",
                out_path,
                "--xml",
                "--feature",
                "psnr",
                "--no_prediction",
            ]

            if self.ref_file.endswith(".yuv"):
                width, height, _ = utils.get_file_desc(os.path.basename(self.ref_file))
                command += [
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "--pixel_format",
                    "420",
                    "--bitdepth",
                    "8",
                ]

            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode != 0:
                self.logger.error(f"Error calculating PSNR: {result.stderr.strip()}")
                return None

            import xml.etree.ElementTree as ET

            tree = ET.parse(out_path)
            vmaf_data = tree.getroot()

            metrics = {}

            # Find all metric elements in pooled_metrics section
            for metric_elem in vmaf_data.findall("./pooled_metrics/metric"):
                name = metric_elem.get("name", "")
                mean = float(metric_elem.get("mean", 0))
                metrics[name] = mean
            return metrics

        except Exception as e:
            self.logger.error(f"Error parsing PSNR output: {e}")
            return None

    @classmethod
    def get_name(cls) -> str:
        return "PSNR"


class QualityMetricsCalculator:
    def __init__(
        self,
        source_png_dir: str,
        source_y4m_dir: str,
        source_yuv_dir: str,
        ref_dir: str,
        y4m_ref_dir: str,
        yuv_ref_dir: str,
        logger: Logger,
        binaries: Dict[str, str],
        allow_metrics: Dict[str, bool],
        aom_ctc_model: str,
        artifacts_path: str = "",
    ):
        self.source_png_dir = source_png_dir
        self.ref_dir = ref_dir
        self.source_y4m_dir = source_y4m_dir
        self.y4m_ref_dir = y4m_ref_dir
        self.source_yuv_dir = source_yuv_dir
        self.yuv_ref_dir = yuv_ref_dir
        self.logger = logger
        self.binaries = binaries
        self.allow_metrics = allow_metrics
        self.aom_ctc_model = aom_ctc_model
        self.metrics: List = []
        self.artifacts_path = artifacts_path
        os.makedirs(self.artifacts_path, exist_ok=True)

        if allow_metrics.get("ssimulacra2", False):
            self.register_metric(
                lambda src, test: SSIMULACRA2Metric(
                    src,
                    test,
                    logger,
                    binaries["ssimulacra2"],
                    self.artifacts_path,
                )
            )

        if allow_metrics.get("vmaf", False) or allow_metrics.get("ms_ssim", False):
            self.register_metric(
                lambda src, test: VMAFMetric(
                    src,
                    test,
                    logger,
                    binaries["vmaf"],
                    self.aom_ctc_model,
                    self.artifacts_path,
                )
            )
        elif allow_metrics.get("psnr", False):
            # special case if VMAF is not enabled, as it computes PSNR already
            self.register_metric(
                lambda src, test: PSNRMetric(
                    src,
                    test,
                    logger,
                    binaries["vmaf"],
                    self.artifacts_path,
                )
            )

    def register_metric(self, metric_factory):
        self.metrics.append(metric_factory)

    def calculate_single_file_metrics(
        self,
        ref_file: str,
        distorted_file: str | None,
        ref_y4m_file: str,
        distorted_y4m_file: str | None,
        ref_yuv_file: str,
        distorted_yuv_file: str | None,
    ) -> Dict[str, float]:
        """Calculate quality metrics for a single pair of files"""
        file_metrics: Dict[str, float] = {}

        # Calculate SSIMULACRA2 if PNG/PGM files are provided
        if (
            distorted_file
            and ref_file
            and distorted_file.lower().endswith((".png", ".pgm"))
            and self.allow_metrics.get("ssimulacra2", False)
        ):
            for metric_factory in self.metrics:
                metric_instance = metric_factory(ref_file, distorted_file)
                if metric_instance.get_name().lower() == "ssimulacra2":
                    value = metric_instance.calculate()
                    if value is not None:
                        file_metrics[metric_instance.get_name().lower()] = value

        # Calculate VMAF/SSIM/MS-SSIM/PSNR if Y4M files are provided
        tmp_distorted_file = (
            distorted_y4m_file if distorted_y4m_file else distorted_yuv_file
        )
        tmp_ref_file = ref_y4m_file if ref_y4m_file else ref_yuv_file
        if (
            tmp_distorted_file
            and tmp_ref_file
            and os.path.exists(tmp_distorted_file)
            and os.path.exists(tmp_ref_file)
            and (
                self.allow_metrics.get("vmaf", False)
                or self.allow_metrics.get("ms_ssim", False)
            )
        ):
            for metric_factory in self.metrics:
                metric_instance = metric_factory(tmp_ref_file, tmp_distorted_file)
                if metric_instance.get_name().lower() == "vmaf":
                    value = metric_instance.calculate()
                    if isinstance(value, dict):
                        for sub_metric_name, sub_value in value.items():
                            file_metrics[sub_metric_name] = sub_value
        elif (
            tmp_distorted_file
            and tmp_ref_file
            and os.path.exists(tmp_distorted_file)
            and os.path.exists(tmp_ref_file)
            and self.allow_metrics.get("psnr", False)
        ):
            for metric_factory in self.metrics:
                metric_instance = metric_factory(tmp_ref_file, tmp_distorted_file)
                if metric_instance.get_name().lower() == "psnr":
                    value = metric_instance.calculate()
                    if isinstance(value, dict):
                        for sub_metric_name, sub_value in value.items():
                            file_metrics[sub_metric_name] = sub_value

        return file_metrics
