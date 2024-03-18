from torch_geometric.data import Data

def pyg2_data_transform(data: Data):
    # convert pyg data to Data class for version compatibility (2.0)
    source = data.__dict__
    if "_store" in source:
        source = source["_store"]
    return Data(**{k: v for k, v in source.items() if v is not None})