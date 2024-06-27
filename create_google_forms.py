from apiclient import discovery
from httplib2 import Http
from oauth2client import client, file, tools
import argparse
import json
from typing import Dict


def create_form(jtbd_json:Dict, title:str):

    SCOPES = "https://www.googleapis.com/auth/forms.body"
    DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

    store = file.Storage("token.json")
    creds = None
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets("gcp_oauth_client_secret.json", SCOPES)
        creds = tools.run_flow(flow, store)

    form_service = discovery.build(
        "forms",
        "v1",
        http=creds.authorize(Http()),
        discoveryServiceUrl=DISCOVERY_DOC,
        static_discovery=False,
    )

    # Request body for creating a form
    NEW_FORM = {
        "info": {
            "title": "Job To Be done",
            "documentTitle": title.title(),
        }
    }

    # Creates the initial form
    result = form_service.forms().create(body=NEW_FORM).execute()


    INFO = {
        "requests": [
            {
                "updateFormInfo": {
                    "info":{
                        "description": jtbd_json["job_statement"]
                    },
                    "updateMask": "description"
                }
            }
            
        ]
    }

    update_setting = (form_service.forms()
        .batchUpdate(formId=result["formId"], body=INFO)
        .execute()
    )

    customer_outcomes = [o["expectation_statement"] 
                                    for o in jtbd_json["outcome_expectations"]
                                    if o["outcome_type"] in ["Customer desired outcome",
                                                                "Customer undesired outcome"]]

    idx = 0

    HEADER = {
            "requests": [
                {
                    "createItem": {
                        "item": {
                            "title": (
                                "Customer Outcome Expectations"
                            ),
                            "description": "Please rate the importance of the following outcomes:",
                            "textItem": {}
                        },
                        "location": {"index": idx}
                    }
                }
            ]
    }
    idx += 1

    text_setting = (form_service.forms()
        .batchUpdate(formId=result["formId"], body=HEADER)
        .execute()
    )
   
    for o in customer_outcomes:

        QUESTION = {
            "requests": [
                {
                    "createItem": {
                        "item": {
                            "title": (
                                f"{o}"
                            ),
                            "questionItem": {
                                "question": {
                                    "required": True,
                                    "scaleQuestion": {
                                        "low": 1,
                                        "high": 4,
                                        "lowLabel": "Not important",
                                        "highLabel": "Very important"
                                    },
                                }
                            },
                        },
                        "location": {"index": idx},
                    }
                }
            ]
        }

        idx += 1

    # Adds the question to the form
        question_setting = (
            form_service.forms()
            .batchUpdate(formId=result["formId"], body=QUESTION)
            .execute()
        )

    # Prints the result to show the question has been added
    get_result = form_service.forms().get(formId=result["formId"]).execute()
    print(get_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', default="scrappy/notebooks/jtbd_0.json")
    parser.add_argument('-t', '--title', default="Financial Payment APIs")

    args = parser.parse_args()
    filename = args.filename
    title = args.title

    with open(filename, "r") as f:
        jtbd_json = json.load(f)

    create_form(jtbd_json, title)
