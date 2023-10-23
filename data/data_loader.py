
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader



def CreateDataLoader_1(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader_1
    data_loader = CustomDatasetDataLoader_1()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataLoader_palette(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader_palette
    data_loader = CustomDatasetDataLoader_palette()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader