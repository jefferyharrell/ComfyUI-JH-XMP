import hashlib
import os
from typing import Any

import folder_paths
import numpy as np
import PIL.Image
import PIL.ImageOps
import PIL.ImageSequence
import torch

from .jh_xmp_metadata import JHXMPMetadata


class JHLoadImageWithXMPMetadataNode:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {"image": (cls.get_image_files(), {"image_upload": True})},
        }

    @classmethod
    def get_image_files(cls) -> list[str]:
        input_dir: str = folder_paths.get_input_directory()
        files: list[str] = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return sorted(files)

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "IMAGE",
        "MASK",
        "creator",
        "title",
        "description",
        "subject",
        "instructions",
        "comment",
        "alt_text",
        "xml_string",
    )
    FUNCTION = "load_image"
    CATEGORY = "XMP Metadata Nodes"
    OUTPUT_NODE = False

    def load_image(
        self, image: str
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        str,
    ]:
        # `image` here is a string, the name of the image file on disk;
        # just the filename, not the full path.
        image_path = folder_paths.get_annotated_filepath(image)

        # This call to PIL.Image.open can raise a variety of exceptions
        # depending on the image format and the state of the file. We
        # deliberately don't catch these exceptions but instead let them
        # propagate up to ComfyUI, which will handle them by displaying
        # an error message to the user.
        #
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
        image_object = PIL.Image.open(image_path)

        first_frame: PIL.Image.Image | None = None
        output_images = []
        output_masks = []

        excluded_formats = ["MPO"]

        xml_string: str = str()
        xmp_metadata = JHXMPMetadata()

        for raw_frame in PIL.ImageSequence.Iterator(image_object):
            if first_frame is None:
                first_frame = raw_frame.copy()

            # Extract XMP metadata from the first frame, if available
            if len(output_images) == 0:
                xmp_data: bytes | str | None = raw_frame.info.get("xmp", None)
                if isinstance(xmp_data, bytes):
                    xml_string = xmp_data.decode("utf-8")
                if xml_string:  # Can't parse None or an empty string
                    xmp_metadata = JHXMPMetadata.from_string(xml_string)

            # Skip frames with different sizes than the first frame
            # (This is pretty much the unlikeliest of all edge cases)
            if raw_frame.size != first_frame.size:
                continue

            # Convert the frame to image and mask tensors
            image_tensor, mask_tensor = self._frame_to_tensors(raw_frame)

            # Append the processed image and mask to the outputs
            output_images.append(image_tensor)
            output_masks.append(mask_tensor.unsqueeze(0))

        # Combine frames into a single tensor if multiple frames exist
        if len(output_images) > 1 and image_object.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (
            output_image,
            output_mask,
            xmp_metadata.creator,
            xmp_metadata.title,
            xmp_metadata.description,
            xmp_metadata.subject,
            xmp_metadata.instructions,
            xmp_metadata.comment,
            xmp_metadata.alt_text,
            xml_string,
        )

    def _frame_to_tensors(
        self, raw_frame: PIL.Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fix image orientation based on EXIF metadata. Do this in
        # place to avoid creating a new image object for each frame.
        PIL.ImageOps.exif_transpose(raw_frame, in_place=True)

        # Convert 32-bit integer images to RGB
        if raw_frame.mode.startswith("I"):
            raw_frame = raw_frame.point(lambda i: i * (1 / 255))
        rgb_frame = raw_frame.convert("RGB")

        # Normalize the image to a tensor with values in [0, 1]
        np_array = np.array(rgb_frame).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(np_array)[None,]

        # Go back to the original raw frame and extract the alpha
        # channel as a mask
        if "A" in raw_frame.getbands():
            np_array = np.array(raw_frame.getchannel("A")).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(np_array)
        else:
            mask_tensor = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return image_tensor, mask_tensor

    @classmethod
    def IS_CHANGED(cls, image: str) -> str:
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image: str) -> str | bool:
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
