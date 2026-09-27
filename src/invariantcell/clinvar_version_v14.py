"""Metadata-only accession-version qualification; classification text excluded."""
from __future__ import annotations
from datetime import date
import re
import xml.etree.ElementTree as ET


def inspect_version(raw: bytes, requested: str):
    if len(raw) > 4 * 1024**2:
        raise ValueError('XML response above qualified intake cap')
    if not re.fullmatch(r'VCV[0-9]{9}\.[1-9][0-9]*', requested):
        raise ValueError('Explicit VCV accession.version required')
    root = ET.fromstring(raw)
    nodes = list(root.iter('VariationArchive'))
    if len(nodes) != 1:
        raise ValueError('Exactly one VariationArchive required')
    node = nodes[0]
    actual = str(node.get('Accession')) + '.' + str(node.get('Version'))
    dates = {}
    for key in ('DateCreated', 'DateLastUpdated', 'MostRecentSubmission'):
        value = node.get(key)
        if value is not None:
            date.fromisoformat(value)
        dates[key] = value
    submissions = []
    for item in node.iter('ClinVarAccession'):
        if item.get('Accession', '').startswith('SCV'):
            submissions.append({k: item.get(k) for k in ('Accession', 'Version', 'DateCreated', 'DateUpdated', 'OrgID')})
    return {'requested': requested, 'returned': actual,
        'status': 'QUALIFIED_VERSION_METADATA' if actual == requested else 'REJECTED_REQUEST_VERSION_MISMATCH',
        'variation_id': node.get('VariationID'), 'variation_name': node.get('VariationName'),
        'record_status': node.findtext('RecordStatus'), 'dates': dates,
        'date_semantics': {'DateCreated': 'VCV record creation, not version publication',
                          'DateLastUpdated': 'record update, possibly unversioned staff metadata',
                          'MostRecentSubmission': 'latest included submission date, not version publication'},
        'scv_version_metadata': submissions,
        'record_version_first_public_at': None, 'biological_event_time': None,
        'classification_values_inspected': False, 'clinical_truth_adjudicated': False,
        'schema_root': root.tag, 'schema_attributes': root.attrib}
