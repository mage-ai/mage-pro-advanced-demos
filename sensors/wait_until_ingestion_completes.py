from mage_ai.orchestration.run_status_checker import check_status

if 'sensor' not in globals():
    from mage_ai.data_preparation.decorators import sensor


@sensor
def check_condition(*args, **kwargs) -> bool:
    """
    Template code for checking if block or pipeline run completed.

    The default polling interval for checking the status of the block run /
    pipeline run is 60 seconds. If you want to adjust the polling interval,
    you can update it in the "Block settings" in the right panel.
    """
    return check_status(
        'ingestion',
        kwargs['execution_date'],
        hours=1,
    )