import json
from nhs_forecast.api import fetch_prescribing_data
from unittest.mock import patch
import urllib.parse
from nhs_forecast.api import fetch_all_prescribing_data
from unittest.mock import call


def test_fetch_prescribing_data_exists():
    assert callable(fetch_prescribing_data)

def test_fake_api_response():
    fake_response = {
        "success": True,
        "result": {
            "records": [
                {
                    "YEAR_MONTH": "2026-07",
                    "BNF_CHEMICAL_SUBSTANCE": "Metformin hydrochloride",
                    "ITEMS": 16,
                },
                {
                    "YEAR_MONTH": "2026-07",
                    "BNF_CHEMICAL_SUBSTANCE": "Metformin hydrochloride",
                    "ITEMS": 2,
                },
            ]
        },
    }

    records = fake_response["result"]["records"]

    assert len(records) == 2
    assert records[0]["ITEMS"] == 16

def test_fetch_prescribing_data_with_mock():
    fake_response = {
        "success": True,
        "result": {
            "records": [
                {"YEAR_MONTH": "2026-07", "ITEMS": 16},
                {"YEAR_MONTH": "2026-07", "ITEMS": 2},
            ]
        },
    }

    with patch("nhs_forecast.api.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = (
            json.dumps(fake_response).encode("utf-8")
        )

        records = fetch_prescribing_data(
            resource_id="EPD_SNOMED_202607",
            bnf_code="0601022B0",
            limit=5,
            offset=10,
        )
        mock_urlopen.assert_called_once()
    called_url = mock_urlopen.call_args.args[0]
   
    parsed_url = urllib.parse.urlparse(called_url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    assert query_params["resource_id"] == ["EPD_SNOMED_202607"]
    assert query_params["limit"] == ["5"]
    assert query_params["offset"] == ["10"]
    
    called_url = mock_urlopen.call_args.args[0]
    print("\nURL requested:", called_url)   
    assert len(records) == 2
    assert records[0]["ITEMS"] == 16
    search_filter = json.loads(query_params["q"][0])

    assert search_filter["BNF_CHEMICAL_SUBSTANCE_CODE"] == "0601022B0"

def test_fetch_all_prescribing_data():
    fake_page_1 = [{"ITEMS": 10}, {"ITEMS": 20}]
    fake_page_2 = [{"ITEMS": 30}, {"ITEMS": 40}]
    fake_page_3 = [{"ITEMS": 50}]

    with patch("nhs_forecast.api.fetch_prescribing_data") as mock_fetch:
        mock_fetch.side_effect = [
            fake_page_1,
            fake_page_2,
            fake_page_3,
        ]

        records = fetch_all_prescribing_data(
            resource_id="EPD_SNOMED_202607",
            bnf_code="0601022B0",
            page_size=2,
        )

    assert len(records) == 5
    assert mock_fetch.call_count == 3
    assert mock_fetch.call_args_list == [
    call(
        resource_id="EPD_SNOMED_202607",
        bnf_code="0601022B0",
        limit=2,
        offset=0,
    ),
    call(
        resource_id="EPD_SNOMED_202607",
        bnf_code="0601022B0",
        limit=2,
        offset=2,
    ),
    call(
        resource_id="EPD_SNOMED_202607",
        bnf_code="0601022B0",
        limit=2,
        offset=4,
    ),
]
