if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import lit


@transformer
def transform(fun_facts_list, *args, **kwargs):
    # fun_facts: list of strings
    # df_salaries: PySpark DataFrame
    spark = SparkSession.builder.getOrCreate()

    # Load the DataFrame from the temporary view created earlier
    df_salaries = spark.sql("SELECT * FROM people_temp_view")
    df_salaries.show()


    # Convert fun_facts list to Spark DataFrame
    fun_facts_schema = StructType([StructField("fun_fact", StringType(), True)])
    fun_facts_rows = [(ff,) for ff in fun_facts_list or []]
    spark_fun_facts = spark.createDataFrame(fun_facts_rows, schema=fun_facts_schema)

    # df_salaries is already a PySpark DataFrame
    spark_salaries = df_salaries

    # Add a key to join (cross join)
    spark_fun_facts = spark_fun_facts.withColumn('key', lit(1))
    spark_salaries = spark_salaries.withColumn('key', lit(1))

    # Cross join
    joined_df = spark_fun_facts.join(spark_salaries, on='key').drop('key')

    return joined_df