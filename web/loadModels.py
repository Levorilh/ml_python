from pathlib import Path
import os


def loadModels(dictionnary):
    models_dir = Path(os.path.realpath(__file__)).parent.parent / 'models'

    for dir_ in models_dir.iterdir():
        if dir_.is_dir():
            for subFile in dir_.iterdir():
                if subFile.is_file() and subFile.name != ".DS_Store":
                    dictionnary[dir_.name].append(subFile.name)
