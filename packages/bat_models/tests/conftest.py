"""Test configuration.

Tests are skipped at module load time when torch / torchvision aren't
installed (each test module top-loads ``pytest.importorskip``). We don't
do a global skip here because it would abort collection for sibling tests
that don't actually need the missing dependency.
"""
