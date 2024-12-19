"""
This module provides the `JHXMPMetadata` class for creating,
manipulating, and parsing XMP (Extensible Metadata Platform) metadata
using the Adobe XMP format.

XMP is a standard for embedding metadata in digital content, such as
images and documents. It is widely used in creative industries to store
information like authors, titles, descriptions, keywords, and editing
instructions. This module is designed to handle such metadata
programmatically in an efficient and easy-to-use manner.

Key Features:
- Create and manage XMP metadata elements such as `creator`, `title`,
  `description`, `subject`, and `instructions`, etc.
- Serialize XMP metadata to a formatted XML string or a complete XMP packet.
- Parse existing XMP metadata from an XML string.
- Uses the `lxml.etree` library for XML processing to ensure compliance with
  XMP specifications.

Namespaces:
The module defines standard namespaces used in XMP metadata, such as:
- `rdf` (Resource Description Framework)
- `dc` (Dublin Core Metadata)
- `xmp` (Adobe XMP Schema)
- `photoshop` (Photoshop-specific fields)
- `exif` (Exchangeable Image File Format)
- `Iptc4xmpCore` (IPTC Core Schema)

References:
- Adobe XMP Specifications:
  https://developer.adobe.com/xmp/docs/XMPSpecifications/

Dependencies:
- `lxml` for XML processing

Example Usage:
```python
from jh_xmp_metadata import JHXMPMetadata

# Create a new XMP metadata object
metadata = JHXMPMetadata()
metadata.creator = "John Doe"
metadata.title = "A Beautiful Sunset"
metadata.description = "A vivid depiction of a sunset over the ocean."
metadata.subject = "sunset, ocean, photography"
metadata.instructions = "Enhance colors slightly."
metadata.comment = "This is a comment."
metadata.alt_text = "A beautiful sunset"

# Convert to XML string
xml_string = metadata.to_string()
print(xml_string)

# Parse existing XMP metadata from XML
parsed_metadata = JHXMPMetadata.from_string(xml_string)
print(parsed_metadata.title)  # Outputs: A Beautiful Sunset"""

# pylint: disable=c-extension-no-member

import re
from typing import Final

from lxml import etree


class JHXMPMetadata:
    NAMESPACES: Final = {
        "x": "adobe:ns:meta/",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "dc": "http://purl.org/dc/elements/1.1/",
        "xml": "http://www.w3.org/XML/1998/namespace",
        "xmp": "http://ns.adobe.com/xap/1.0/",
        "photoshop": "http://ns.adobe.com/photoshop/1.0/",
        "exif": "http://ns.adobe.com/exif/1.0/",
        "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
    }

    def __init__(self):
        self._creator: str | None = None
        self._title: str | None = None
        self._description: str | None = None
        self._subject: str | None = None
        self._instructions: str | None = None
        self._comment: str | None = None
        self._alt_text: str | None = None

        # Set up the empty XMP metadata tree. We will add (and remove) elements as needed.
        self._xmpmetadata = etree.Element(
            "{adobe:ns:meta/}xmpmeta", nsmap=self.NAMESPACES
        )
        self._xmpmetadata.set(
            "{adobe:ns:meta/}xmptk",
            "Adobe XMP Core 6.0-c002 79.164861, 2016/09/14-01:09:01",
        )
        self._rdf = etree.SubElement(
            self._xmpmetadata, "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF"
        )
        self._rdf_description = etree.SubElement(
            self._rdf,
            "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description",
            attrib={"{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about": ""},
        )
        self._dc_creator_element = None
        self._dc_title_element = None
        self._dc_description_element = None
        self._dc_subject_element = None
        self._photoshop_instructions_element = None
        self._exif_usercomment_element = None
        self._Iptc4xmpCore_alt_text_element = None

    @property
    def creator(self) -> str | None:
        return self._creator

    @creator.setter
    def creator(self, value: str | None) -> None:
        if value is None or value == "" or value.strip() == "":
            self._creator = None
            if self._dc_creator_element is not None:
                self._rdf_description.remove(self._dc_creator_element)
        else:
            self._creator = value
            _creators = self._string_to_list(self._creator)
            self._dc_creator_element = etree.SubElement(
                self._rdf_description, "{http://purl.org/dc/elements/1.1/}creator"
            )
            _seq = etree.SubElement(
                self._dc_creator_element,
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Seq",
            )
            for _creator in _creators:
                _li = etree.SubElement(
                    _seq,
                    "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li",
                    attrib={"{http://www.w3.org/XML/1998/namespace}lang": "x-default"},
                )
                _li.text = _creator

    @property
    def title(self) -> str | None:
        return self._title

    @title.setter
    def title(self, value: str | None) -> None:
        if value is None or value == "" or value.strip() == "":
            self._title = None
            if self._dc_title_element is not None:
                self._rdf_description.remove(self._dc_title_element)
        else:
            self._title = value
            self._dc_title_element = etree.SubElement(
                self._rdf_description, "{http://purl.org/dc/elements/1.1/}title"
            )
            _alt = etree.SubElement(
                self._dc_title_element,
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Alt",
            )
            _li = etree.SubElement(
                _alt,
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li",
                attrib={"{http://www.w3.org/XML/1998/namespace}lang": "x-default"},
            )
            _li.text = self._title

    @property
    def description(self) -> str | None:
        return self._description

    @description.setter
    def description(self, value: str | None) -> None:
        if value is None or value == "" or value.strip() == "":
            self._description = None
            if self._dc_description_element is not None:
                self._rdf_description.remove(self._dc_description_element)
        else:
            self._description = value
            self._dc_description_element = etree.SubElement(
                self._rdf_description, "{http://purl.org/dc/elements/1.1/}description"
            )
            _alt = etree.SubElement(
                self._dc_description_element,
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Alt",
            )
            _li = etree.SubElement(
                _alt,
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li",
                attrib={"{http://www.w3.org/XML/1998/namespace}lang": "x-default"},
            )
            _li.text = self._description

    @property
    def subject(self) -> str | None:
        return self._subject

    @subject.setter
    def subject(self, value: str | None) -> None:
        if value is None or value == "" or value.strip() == "":
            self._subject = None
            if self._dc_subject_element is not None:
                self._rdf_description.remove(self._dc_subject_element)
        else:
            self._subject = value
            _subjects = self._string_to_list(self._subject)
            self._dc_subject_element = etree.SubElement(
                self._rdf_description, "{http://purl.org/dc/elements/1.1/}subject"
            )
            _bag = etree.SubElement(
                self._dc_subject_element,
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Bag",
            )
            for _subject in _subjects:
                _li = etree.SubElement(
                    _bag,
                    "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li",
                    attrib={"{http://www.w3.org/XML/1998/namespace}lang": "x-default"},
                )
                _li.text = _subject

    @property
    def instructions(self) -> str | None:
        return self._instructions

    @instructions.setter
    def instructions(self, value: str | None) -> None:
        if value is None or value == "" or value.strip() == "":
            self._instructions = None
            if self._photoshop_instructions_element is not None:
                self._rdf_description.remove(self._photoshop_instructions_element)
        else:
            self._instructions = value
            self._photoshop_instructions_element = etree.SubElement(
                self._rdf_description,
                "{http://ns.adobe.com/photoshop/1.0/}Instructions",
            )
            self._photoshop_instructions_element.text = self._instructions

    @property
    def comment(self) -> str | None:
        return self._comment

    @comment.setter
    def comment(self, value: str | None) -> None:
        if value is None or value == "" or value.strip() == "":
            self._comment = None
            if self._exif_usercomment_element is not None:
                self._rdf_description.remove(self._exif_usercomment_element)
        else:
            self._comment = value
            self._exif_usercomment_element = etree.SubElement(
                self._rdf_description,
                "{http://ns.adobe.com/exif/1.0/}UserComment",
            )
            _alt = etree.SubElement(
                self._exif_usercomment_element,
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Alt",
            )
            _li = etree.SubElement(
                _alt,
                "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li",
                attrib={"{http://www.w3.org/XML/1998/namespace}lang": "x-default"},
            )
            _li.text = self._comment

    @property
    def alt_text(self) -> str | None:
        return self._alt_text

    @alt_text.setter
    def alt_text(self, value: str | None) -> None:
        if value is None or value == "" or value.strip() == "":
            self._alt_text = None
            if self._Iptc4xmpCore_alt_text_element is not None:
                self._rdf_description.remove(self._Iptc4xmpCore_alt_text_element)
        else:
            self._alt_text = value
            self._Iptc4xmpCore_alt_text_element = etree.SubElement(
                self._rdf_description,
                "{http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/}AltTextAccessibility",
            )
            self._Iptc4xmpCore_alt_text_element.text = self._alt_text

    def _string_to_list(self, string: str) -> list[str]:
        return re.split(r"[;,]\s*", string)

    def to_string(self, pretty_print: bool = True) -> str:
        return etree.tostring(
            self._xmpmetadata, pretty_print=pretty_print, encoding="UTF-8"
        ).decode("utf-8")

    def to_wrapped_string(self) -> str:
        return f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>{self.to_string()}<?xpacket end="w"?>"""  # pylint: disable=line-too-long

    @classmethod
    def from_string(cls, xml_string: str) -> "JHXMPMetadata":
        instance = cls()

        try:
            root = etree.fromstring(xml_string)
        except etree.XMLSyntaxError:
            # raise ValueError("Invalid XML") from e
            # In case of invalid XML, return an empty instance
            return instance

        creator_list: list[str] = list()
        dc_creator_element = root.xpath(
            "//dc:creator/rdf:Seq/rdf:li", namespaces=cls.NAMESPACES
        )
        if len(dc_creator_element) > 0:
            for creator in dc_creator_element:
                creator_list.append(creator.text)
            instance.creator = ", ".join(creator_list)

        dc_title_element = root.xpath(
            "//dc:title/rdf:Alt/rdf:li", namespaces=cls.NAMESPACES
        )
        if len(dc_title_element) > 0:
            instance.title = dc_title_element[0].text

        dc_description_element = root.xpath(
            "//dc:description/rdf:Alt/rdf:li", namespaces=cls.NAMESPACES
        )
        if len(dc_description_element) > 0:
            instance.description = dc_description_element[0].text

        dc_subject_element = root.xpath(
            "//dc:subject/rdf:Bag/rdf:li", namespaces=cls.NAMESPACES
        )
        if len(dc_subject_element) > 0:
            subject_list: list[str] = list()
            for subject in dc_subject_element:
                subject_list.append(subject.text)
            instance.subject = ", ".join(subject_list)

        photoshop_instructions_element = root.xpath(
            "//photoshop:Instructions", namespaces=cls.NAMESPACES
        )
        if len(photoshop_instructions_element) > 0:
            instance.instructions = photoshop_instructions_element[0].text

        dc_comment_element = root.xpath(
            "//exif:UserComment/rdf:Alt/rdf:li", namespaces=cls.NAMESPACES
        )
        if len(dc_comment_element) > 0:
            instance.comment = dc_comment_element[0].text

        Iptc4xmpCore_alt_text_element = root.xpath(
            "//Iptc4xmpCore:AltTextAccessibility", namespaces=cls.NAMESPACES
        )
        if len(Iptc4xmpCore_alt_text_element) > 0:
            instance.alt_text = Iptc4xmpCore_alt_text_element[0].text

        return instance
