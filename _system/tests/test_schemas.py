"""Tests for Pydantic schemas and status helpers."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from _system.schemas.entities import Entity, EntityType, PaperEntities
from _system.schemas.figures import PaperFigure
from _system.schemas.paper_metadata import (
    SECTION_CHUNK_LEVELS,
    STATUS_ORDER,
    PaperMetadata,
    PaperStatus,
    can_run_from,
)
from _system.schemas.taxonomy import ClassificationOutput


class TestPaperStatus:
    def test_has_six_members(self):
        members = list(PaperStatus)
        assert len(members) == 6
        assert PaperStatus.FETCHED in members
        assert PaperStatus.CONVERTED in members
        assert PaperStatus.CLASSIFIED in members
        assert PaperStatus.EXTRACTED in members
        assert PaperStatus.INDEXED in members
        assert PaperStatus.FAILED_HTML in members

    def test_values_are_lowercase_strings(self):
        assert PaperStatus.FETCHED.value == "fetched"
        assert PaperStatus.CONVERTED.value == "converted"
        assert PaperStatus.CLASSIFIED.value == "classified"
        assert PaperStatus.EXTRACTED.value == "extracted"
        assert PaperStatus.INDEXED.value == "indexed"
        assert PaperStatus.FAILED_HTML.value == "failed_html"


class TestStatusOrder:
    def test_distinct_ordinal_for_each_status(self):
        ordinals = list(STATUS_ORDER.values())
        assert len(ordinals) == 6
        assert len(set(ordinals)) == 6

    def test_failed_html_is_minus_one(self):
        assert STATUS_ORDER[PaperStatus.FAILED_HTML] == -1

    def test_real_stages_ascend_from_zero(self):
        assert STATUS_ORDER[PaperStatus.FETCHED] == 0
        assert STATUS_ORDER[PaperStatus.CONVERTED] == 1
        assert STATUS_ORDER[PaperStatus.CLASSIFIED] == 2
        assert STATUS_ORDER[PaperStatus.EXTRACTED] == 3
        assert STATUS_ORDER[PaperStatus.INDEXED] == 4


class TestCanRunFrom:
    def test_none_current_to_fetched_is_true(self):
        assert can_run_from(None, PaperStatus.FETCHED) is True

    def test_none_current_to_any_target_is_true(self):
        for target in PaperStatus:
            assert can_run_from(None, target) is True

    def test_fetched_to_converted_is_true(self):
        assert can_run_from(PaperStatus.FETCHED, PaperStatus.CONVERTED) is True

    def test_rerun_current_stage_is_true(self):
        assert can_run_from(PaperStatus.CONVERTED, PaperStatus.CONVERTED) is True

    def test_extracted_to_fetched_is_false(self):
        assert can_run_from(PaperStatus.EXTRACTED, PaperStatus.FETCHED) is False

    def test_cannot_skip_ahead(self):
        assert can_run_from(PaperStatus.FETCHED, PaperStatus.CLASSIFIED) is False
        assert can_run_from(PaperStatus.FETCHED, PaperStatus.INDEXED) is False

    def test_failed_html_never_proceeds(self):
        for target in PaperStatus:
            assert can_run_from(PaperStatus.FAILED_HTML, target) is False


class TestPaperMetadata:
    def _minimal_kwargs(self) -> dict:
        return dict(
            arxiv_id="2401.12345",
            paper_name="SomeName2024",
            title="A great paper",
            authors="A, B, C",
            date="2024-01-15",
            abstract="summary",
            pdf_url="https://arxiv.org/pdf/2401.12345",
            status=PaperStatus.FETCHED,
        )

    def test_accepts_minimal_required_fields(self):
        meta = PaperMetadata(**self._minimal_kwargs())
        assert meta.arxiv_id == "2401.12345"
        # use_enum_values=True stores the raw string, not the enum member.
        assert meta.status == "fetched"

    def test_markdown_and_raw_html_optional(self):
        meta = PaperMetadata(**self._minimal_kwargs())
        assert meta.markdown is None
        assert meta.raw_html is None

        kwargs = self._minimal_kwargs()
        kwargs.update(markdown="# Hello", raw_html="<html></html>")
        meta2 = PaperMetadata(**kwargs)
        assert meta2.markdown == "# Hello"
        assert meta2.raw_html == "<html></html>"

    def test_needs_review_defaults_false(self):
        meta = PaperMetadata(**self._minimal_kwargs())
        assert meta.needs_review is False

    def test_domain_and_collection_optional(self):
        meta = PaperMetadata(**self._minimal_kwargs())
        assert meta.domain is None
        assert meta.collection is None

    def test_rejects_missing_required_field(self):
        kwargs = self._minimal_kwargs()
        del kwargs["arxiv_id"]
        with pytest.raises(ValidationError):
            PaperMetadata(**kwargs)

    def test_rejects_missing_date(self):
        kwargs = self._minimal_kwargs()
        del kwargs["date"]
        with pytest.raises(ValidationError):
            PaperMetadata(**kwargs)

    def test_rejects_missing_pdf_url(self):
        kwargs = self._minimal_kwargs()
        del kwargs["pdf_url"]
        with pytest.raises(ValidationError):
            PaperMetadata(**kwargs)

    def test_rejects_bogus_status_string(self):
        kwargs = self._minimal_kwargs()
        kwargs["status"] = "not_a_real_status"
        with pytest.raises(ValidationError):
            PaperMetadata(**kwargs)


class TestEntityType:
    def test_has_six_members(self):
        members = list(EntityType)
        assert len(members) == 6
        expected = {"method", "dataset", "metric", "model", "technique", "benchmark"}
        assert {m.value for m in members} == expected


class TestEntity:
    def test_minimal_entity_has_default_aliases_and_description(self):
        e = Entity(name="BERT", type=EntityType.MODEL, source_section="# Method")
        assert e.name == "BERT"
        assert e.aliases == []
        assert e.description is None

    def test_paper_entities_holds_entity_list(self):
        pe = PaperEntities(
            paper_name="P",
            domain="nlp",
            entities=[
                Entity(name="X", type=EntityType.METHOD, source_section="# M"),
            ],
        )
        assert len(pe.entities) == 1
        assert pe.entities[0].type == EntityType.METHOD


class TestPaperFigure:
    def test_round_trips_image_bytes_and_mime(self):
        img = b"\x89PNG\r\n\x1a\nsomething"
        fig = PaperFigure(
            figure_number=1,
            figure_id="S1.F1",
            caption="Test caption",
            section_context="# Results",
            image_data=img,
            mime_type="image/png",
        )
        assert fig.image_data == img
        assert fig.mime_type == "image/png"
        assert fig.display_number is None

    def test_display_number_accepts_caption_label(self):
        fig = PaperFigure(
            figure_number=3,
            figure_id="S3.F1",
            caption="Fig 3a",
            section_context="# Discussion",
            image_data=b"\x00",
            mime_type="image/jpeg",
            display_number="3a",
        )
        assert fig.display_number == "3a"


class TestClassificationOutput:
    def test_validates_full_payload(self):
        payload = {
            "domain": "nlp",
            "domain_is_new": False,
            "collection": "transformers",
            "topics": ["attention", "pretraining"],
        }
        co = ClassificationOutput.model_validate(payload)
        assert co.domain == "nlp"
        assert co.domain_is_new is False
        assert co.collection == "transformers"
        assert co.topics == ["attention", "pretraining"]

    def test_accepts_domain_is_new_true_with_empty_topics(self):
        payload = {
            "domain": "new-field",
            "domain_is_new": True,
            "collection": "c",
            "topics": [],
        }
        co = ClassificationOutput.model_validate(payload)
        assert co.domain_is_new is True
        assert co.topics == []

    def test_rejects_missing_domain(self):
        with pytest.raises(ValidationError):
            ClassificationOutput.model_validate(
                {"domain_is_new": False, "collection": "c", "topics": []}
            )

    def test_forbids_extras(self):
        with pytest.raises(ValidationError):
            ClassificationOutput.model_validate(
                {
                    "domain": "nlp",
                    "domain_is_new": False,
                    "collection": "c",
                    "topics": [],
                    "confidence": 0.9,
                }
            )


class TestSectionChunkLevels:
    def test_is_header_levels_1_through_3(self):
        assert SECTION_CHUNK_LEVELS == (1, 2, 3)


class TestSchemaImportsStayLight:
    """Schemas must not drag in torch/sentence_transformers/gliner2.

    search.py depends on schema modules and must stay under 300 ms for --help.
    """

    def test_schema_modules_do_not_import_heavy_deps(self):
        import importlib
        import sys

        # Drop any preloaded copies so we see what actually gets pulled in
        # as a side effect of a cold import.
        shed = [
            m
            for m in sys.modules
            if m.startswith(
                (
                    "_system.schemas",
                    "_system.utils.config",
                    "sentence_transformers",
                    "gliner2",
                    "torch",
                )
            )
        ]
        for m in shed:
            del sys.modules[m]

        for mod in (
            "_system.schemas.paper_metadata",
            "_system.schemas.entities",
            "_system.schemas.figures",
            "_system.schemas.taxonomy",
            "_system.utils.config",
        ):
            importlib.import_module(mod)

        forbidden = [
            m
            for m in sys.modules
            if m.startswith(("sentence_transformers", "gliner2", "torch"))
        ]
        assert forbidden == [], f"heavy deps leaked into schema imports: {forbidden}"
