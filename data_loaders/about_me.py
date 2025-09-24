if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    about_me = kwargs.get('about_me', dict(
        bio='I am a magic user',
    ))

    return about_me