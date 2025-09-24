if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from pyspark.sql import Row, SparkSession


@data_loader
def load_data(*args, **kwargs):
    spark = SparkSession.builder.getOrCreate()
    data = [
        Row(id=1, name="Alice", age=29, salary=50000),
        Row(id=2, name="Bob", age=35, salary=60000),
        Row(id=3, name="Charlie", age=40, salary=70000),
        Row(id=4, name="David", age=45, salary=None)  # Null value example
    ]

    df = spark.createDataFrame(data)
    df.show()

    # Write DataFrame back to Spark (e.g., as a table or file)
    # Example: write to a temporary view
    df.createOrReplaceTempView("people_temp_view")

    # Example: write to parquet (uncomment if you want to write to disk)
    df.write.mode("overwrite").parquet("/tmp/people.parquet")

    return df