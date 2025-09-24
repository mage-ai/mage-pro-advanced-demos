import polars as pl

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    df = pl.from_pandas(data['load_api_data_from_ingestion'][0]).unique(subset=['_name'])

    about_me = kwargs.get('about_me', dict(
        bio='I am a magic user',
    ))

    return dict(
        sample_data=df.to_dicts(),
        about_me=about_me,
    )