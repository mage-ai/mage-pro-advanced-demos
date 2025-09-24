@transformer
def transform(df, *args, **kwargs):
    df['employeeCount'] = df['employeeCount'].apply(lambda x: round(x))
    return df