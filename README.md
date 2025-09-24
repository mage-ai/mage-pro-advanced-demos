# Mage Pro Advanced Use Cases

![Pipeline Graph](./pipeline_graph.png)

## Ingestion workflow

Schedule: every 15 minutes

### Steps:

1. Load data from API
1. Transform by rounding the employeeCount column
1. Export data to DuckDB

## SQL + Python workflow

Schedule: Hourly after waiting for ingestion to complete

### Steps:

1. Sensor: wait until ingestion workflow completes
1. Load titanic dataset from CSV
1. Load ingested data from DuckDB using SQL block and join with the data from the Python block that loads from the CSV
1. Trigger dynamic block workflow: this workflow passes in runtime variables when it triggers the dynamic block workflow; that value is available throughout the dynamic block workflow via the keyword arguments (e.g. kwargs)
1. Trigger ML training workflow

## Dynamic block workflows

Triggered by SQL + Python workflow

### Steps:

1. Dynamic block uses the variable passed from the SQL + Python workflow (e.g. count), then spawn N child blocks where N is the count variable value divided by 1000.
1. Each child block will received 1 item from the list produced by the 1st block; for example, if the 1st block returns a list of 10 items, the downstream child block will be spawned 10 times
1. AI block: using the data passed from the upstream block, pass that data into an LLM and prompt the model. 1. After fanning out N times, reduce all the outputs into a single list and pass that data to the next block
1. Trigger Spark workflow: use the collected list of items from the AI blocks and trigger the Spark workflow and pass that data as a runtime variable

## Spark workflow

Triggered by Dynamic block workflow

### Steps:

1. Generate sample salary data and write to Spark
1. Load fun facts from runtime variables via the "fun_facts" keyword argument that’s passed to the workflow by the Dynamic block workflow
1. Transform: combine the data from the 1st two blocks

## ML training and inference

Triggered by SQL + Python workflow

### Steps:

1.  Global Data Product (this single data product can be re-used across workflows as a single canonical dataset that only recomputes if it’s stale e.g. after 1 day seconds. If the dataset is not stale, the most recent computed data will be used): Load core user data from
1.  Transform: prepare the core user data for training
1.  Train ML model: train XGBoost model to predict "survived" column
1.  Online inference block to be used to predict whether a person survives based on a set of features

    **Example API request**:

         curl -X POST https://cluster.mage.ai/mageai-0-demo/api/runs \
         --header 'Content-Type: application/json' \
         --header 'Authorization: Bearer XXXXXXXXXXXX' \
         --data '
         {
             "run": {
                 "pipeline_uuid": "ml_training_and_inference",
                 "block_uuid": "online_inference",
                 "variables": {
                 "pclass": "3",
                 "_name": "Kim",
                 "sex": "female",
                 "age": "21",
                 "sibsp": "0",
                 "parch": "0",
                 "ticket": "315037",
                 "fare": "100",
                 "cabin": "D21",
                 "embarked": "S"
                 }
             }
         }'

1.  AI block: given the person’s attributes and survival prediction, get an explanation as to why that prediction was made

## LLM inference

API triggered endpoint

### Steps:

1.  Global data product: use the canonical core users data
1.  Transform: de-dupe
1.  AI block: analyze and create a decision making framework for deciding if someone survives based on any information they provide
1.  AI block: predict and explain why someone would survive a boat crash based on any runtime variable data they provide about a person

    **Example API request**:

         curl -X POST https://cluster.mage.ai/mageai-0-demo/api/runs \
             --header 'Content-Type: application/json' \
             --header 'Authorization: Bearer XXXXXXXXXXXX' \
             --data '
             {
                 "run": {
                     "pipeline_uuid": "llm_inference",
                     "block_uuid": "will_i_survive",
                     "variables": {
                     "about_me": {
                         "bio": "I am a 21-year-old male from Japan and I have OP cheat-magic. I am a student and I love to travel."
                     }
                     }
                 }
             }'
