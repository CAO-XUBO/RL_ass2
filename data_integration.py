import pandas as pd

def data_load(data_filepath):
    df = pd.read_csv(data_filepath)
    return df

def data_integration(location_data, range_data):
    location_data['range'] = range_data['range']
    return location_data

def data_output(df, output_filepath):
    df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    location_filepath = "origin_data/robot_locations.csv"
    range_filepath = "origin_data/range.csv"

    output_filepath = "processed_data/robot_locations_range.csv"

    location_data = data_load(location_filepath)
    range_data = data_load(range_filepath)

    integration_df = data_integration(location_data, range_data)

    data_output(integration_df, output_filepath)
