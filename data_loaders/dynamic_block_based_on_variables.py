if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    count = kwargs.get('count', 10000)

    print(f'Count from pipeline that triggered me: {count}')

    count = int(count / 1000)

    child_blocks = [i for i in range(count)]

    return [
        child_blocks,
    ]