import pytest
from lxml import etree

from ..comfyui_jh_xmp_metadata_nodes.jh_xmp_metadata import JHXMPMetadata


@pytest.fixture
def metadata() -> JHXMPMetadata:
    return JHXMPMetadata()


@pytest.fixture
def example_metadata() -> dict[str, str]:
    return {
        "creator": "Jane Doe",
        "title": "A Beautiful Sunset",
        "description": "A vivid depiction of a sunset over the ocean.",
        "subject": "sunset, ocean, photography",
        "instructions": "Enhance colors slightly.",
        "comment": "This is a comment.",
        "alt_text": "A beautiful sunset",
    }


@pytest.fixture
def example_metadata_with_unicode() -> dict[str, str]:
    return {
        "creator": "Café à la mode 😊",
        "title": "一個美麗的夕陽",
        "description": "A vivid depiction of a sunset 🌅 over the ocean.",
        "subject": "🌅, ocean, photography",
        "instructions": "Enhance colors slightly 💯.",
        "comment": "This is a comment. 😎",
        "alt_text": "A beautiful sunset 🌅",
    }


@pytest.fixture
def populated_metadata(example_metadata: dict[str, str]) -> JHXMPMetadata:
    metadata = JHXMPMetadata()
    metadata.creator = example_metadata["creator"]
    metadata.title = example_metadata["title"]
    metadata.description = example_metadata["description"]
    metadata.subject = example_metadata["subject"]
    metadata.instructions = example_metadata["instructions"]
    metadata.comment = example_metadata["comment"]
    metadata.alt_text = example_metadata["alt_text"]
    return metadata


def test_initialization(metadata: JHXMPMetadata) -> None:
    assert metadata.creator is None
    assert metadata.title is None
    assert metadata.description is None
    assert metadata.subject is None
    assert metadata.instructions is None
    assert metadata.comment is None
    assert metadata.alt_text is None


def test_to_string_from_empty(metadata: JHXMPMetadata) -> None:
    xml = metadata.to_string()
    try:
        etree.fromstring(xml, parser=etree.XMLParser(recover=True))
    except etree.XMLSyntaxError as e:
        pytest.fail(f"Generated XML is invalid: {e}")


@pytest.mark.parametrize(
    "field_name",
    [
        "creator",
        "title",
        "description",
        "subject",
        "instructions",
        "comment",
        "alt_text",
    ],
)
def test_field_setter_getter(
    metadata: JHXMPMetadata, example_metadata: dict[str, str], field_name: str
):
    setattr(metadata, field_name, example_metadata[field_name])
    assert getattr(metadata, field_name) == example_metadata[field_name]
    setattr(metadata, field_name, None)
    assert getattr(metadata, field_name) is None


def validate_xml_against_metadata(xml: str, populated_metadata: JHXMPMetadata):
    root = etree.fromstring(xml, parser=etree.XMLParser(recover=True))

    def validate_field(xpath: str, expected_value: str | None, field_name: str):
        elements = root.xpath(xpath, namespaces=JHXMPMetadata.NAMESPACES)
        if expected_value is None:
            assert not elements, f"{field_name} should not be in XML"
        else:
            assert (
                len(elements) == 1
            ), f"Expected one {field_name}, found {len(elements)}"
            assert elements[0].text == expected_value, f"{field_name} mismatch"

    validate_field("//dc:creator/rdf:Seq/rdf:li", populated_metadata.creator, "Creator")
    validate_field("//dc:title/rdf:Alt/rdf:li", populated_metadata.title, "Title")
    validate_field(
        "//dc:description/rdf:Alt/rdf:li", populated_metadata.description, "Description"
    )

    subject_li_list = root.xpath(
        "//dc:subject/rdf:Bag/rdf:li", namespaces=JHXMPMetadata.NAMESPACES
    )
    assert len(subject_li_list) == 3
    for li in subject_li_list:
        assert li.text in populated_metadata.subject

    validate_field(
        "//photoshop:Instructions", populated_metadata.instructions, "Instructions"
    )

    validate_field(
        "//exif:UserComment/rdf:Alt/rdf:li", populated_metadata.comment, "Comment"
    )

    validate_field(
        "//Iptc4xmpCore:AltTextAccessibility", populated_metadata.alt_text, "Alt Text"
    )


def test_to_string(populated_metadata: JHXMPMetadata):
    validate_xml_against_metadata(populated_metadata.to_string(), populated_metadata)


def test_to_wrapped_string(populated_metadata: JHXMPMetadata):
    validate_xml_against_metadata(
        populated_metadata.to_wrapped_string(), populated_metadata
    )


def test_from_string(populated_metadata: JHXMPMetadata):
    xml_string = populated_metadata.to_string()
    parsed_metadata = JHXMPMetadata.from_string(xml_string)
    assert parsed_metadata.creator == populated_metadata.creator
    assert parsed_metadata.title == populated_metadata.title
    assert parsed_metadata.description == populated_metadata.description
    assert parsed_metadata.subject == populated_metadata.subject
    assert parsed_metadata.instructions == populated_metadata.instructions
    assert parsed_metadata.comment == populated_metadata.comment
    assert parsed_metadata.alt_text == populated_metadata.alt_text


def test_from_string_with_garbage_data():
    garbage_data = """
    <x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:dc="http://purl.org/dc/elements/1.1/">
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="">
                <dc:title>
                    <rdf:Alt>
                        <rdf:li xml:lang="x-default">A Beautiful Sunset</rdf:li>
                    </rdf:Alt>
                </title>
            </rdf:Description>
        </rdf:RDF>
    </x:xmpmeta>
    """
    parsed_metadata = JHXMPMetadata.from_string(garbage_data)
    assert parsed_metadata.title is None
    assert parsed_metadata.creator is None
    assert parsed_metadata.description is None
    assert parsed_metadata.subject is None
    assert parsed_metadata.instructions is None
    assert parsed_metadata.comment is None
    assert parsed_metadata.alt_text is None


def test_from_string_with_missing_fields():
    xml_string = """
    <x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:dc="http://purl.org/dc/elements/1.1/">
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="">
                <dc:title>
                    <rdf:Alt>
                        <rdf:li xml:lang="x-default">A Beautiful Sunset</rdf:li>
                    </rdf:Alt>
                </dc:title>
            </rdf:Description>
        </rdf:RDF>
    </x:xmpmeta>
    """
    parsed_metadata = JHXMPMetadata.from_string(xml_string)
    assert parsed_metadata.title == "A Beautiful Sunset"
    assert parsed_metadata.creator is None
    assert parsed_metadata.description is None
    assert parsed_metadata.subject is None
    assert parsed_metadata.instructions is None
    assert parsed_metadata.comment is None
    assert parsed_metadata.alt_text is None


def test_large_metadata_values():
    large_string = "A" * 10000
    metadata = JHXMPMetadata()
    metadata.title = large_string
    xml_string = metadata.to_string()
    parsed_metadata = JHXMPMetadata.from_string(xml_string)
    assert parsed_metadata.title == large_string


def test_empty_xml_string():
    empty_xml = ""
    parsed_metadata = JHXMPMetadata.from_string(empty_xml)
    assert parsed_metadata.title is None
    assert parsed_metadata.creator is None
    assert parsed_metadata.description is None
    assert parsed_metadata.subject is None
    assert parsed_metadata.instructions is None
    assert parsed_metadata.comment is None
    assert parsed_metadata.alt_text is None


def test_special_characters_in_xml(
    metadata: JHXMPMetadata, example_metadata_with_unicode: dict[str, str]
):
    for key, value in example_metadata_with_unicode.items():
        setattr(metadata, key, value)
    xml_string = metadata.to_string()
    parsed_metadata = JHXMPMetadata.from_string(xml_string)
    for key, value in example_metadata_with_unicode.items():
        assert getattr(parsed_metadata, key) == value
