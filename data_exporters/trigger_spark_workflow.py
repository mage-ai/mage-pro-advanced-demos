from mage_ai.orchestration.triggers.api import trigger_pipeline
if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def trigger(fun_facts, *args, **kwargs):
    for idx, fun_fact in enumerate(fun_facts):
        print(f'{idx}. {fun_fact}')

    trigger_pipeline(
        'spark_workflow',        # Required: enter the UUID of the pipeline to trigger
        variables={
            'fun_facts': fun_facts,
        },           # Optional: runtime variables for the pipeline
        check_status=False,     # Optional: poll and check the status of the triggered pipeline
        error_on_failure=False, # Optional: if triggered pipeline fails, raise an exception
        poll_interval=60,       # Optional: check the status of triggered pipeline every N seconds
        poll_timeout=None,      # Optional: raise an exception after N seconds
        verbose=True,           # Optional: print status of triggered pipeline run
    )