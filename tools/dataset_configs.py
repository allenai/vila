class GROTOAP2:
    _name = "GROTOAP2".lower()
    train_file = "../data/grotoap2/train-token.json"
    validation_file = "../data/grotoap2/dev-token.json"
    test_file = "../data/grotoap2/test-token.json"
    label_map_file = "../data/grotoap2/labels.json"


class DocBank:
    _name = "DocBank".lower()
    train_file = "../data/docbank/train-token.json"
    validation_file = "../data/docbank/dev-token.json"
    test_file = "../data/docbank/test-token.json"
    label_map_file = "../data/docbank/labels.json"

def instiantiate_dataset(dataset_name):
    if dataset_name.lower() == GROTOAP2._name:
        return GROTOAP2()
    elif dataset_name.lower() == DocBank._name:
        return DocBank()