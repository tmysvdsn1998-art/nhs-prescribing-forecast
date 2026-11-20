import json
import urllib.parse
import urllib.request


BASE_URL = "https://opendata.nhsbsa.net/api/3/action/datastore_search"


def fetch_prescribing_data(resource_id, bnf_code, limit=5, offset=0):
    params = {
        "resource_id": resource_id,
        "limit": limit,
        "offset": offset,
        "q": json.dumps({
            "BNF_CHEMICAL_SUBSTANCE_CODE": bnf_code
        }),
    }

    url = BASE_URL + "?" + urllib.parse.urlencode(params)

    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read().decode("utf-8")
        result = json.loads(data)

    return result["result"]["records"]

def fetch_all_prescribing_data(resource_id, bnf_code, page_size=100):
    all_records = []
    offset = 0

    while True:
        records = fetch_prescribing_data(
            resource_id=resource_id,
            bnf_code=bnf_code,
            limit=page_size,
            offset=offset,
        )

        all_records.extend(records)

        if len(records) < page_size:
            break

        offset += page_size

    return all_records
    