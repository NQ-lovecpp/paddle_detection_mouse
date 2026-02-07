import os
import os.path as osp
import re
import random

devkit_dir = './'
years = ['2007', '2012']


def get_dir(devkit_dir,  type):
    return osp.join(devkit_dir, type)


def walk_dir(devkit_dir):
    filelist_dir = get_dir(devkit_dir, 'ImageSets/Main')
    annotation_dir = get_dir(devkit_dir, 'annotations')
    img_dir = get_dir(devkit_dir, 'images')
    trainval_list = []
    test_list = []
    added = set()

    for _, _, files in os.walk(filelist_dir):
        for fname in files:
            img_ann_list = []
            if re.match('train\.txt', fname):
                img_ann_list = trainval_list
            elif re.match('val\.txt', fname):
                img_ann_list = test_list
            else:
                continue
            fpath = osp.join(filelist_dir, fname)
            for line in open(fpath):
                name_prefix = line.strip().split()[0]
                if name_prefix in added:
                    continue
                added.add(name_prefix)
                ann_path = osp.join(annotation_dir, name_prefix + '.xml')
                img_path = osp.join(img_dir, name_prefix + '.jpg')
                assert os.path.isfile(ann_path), 'file %s not found.' % ann_path
                assert os.path.isfile(img_path), 'file %s not found.' % img_path
                img_ann_list.append((img_path, ann_path))

    return trainval_list, test_list


def prepare_filelist(devkit_dir, output_dir):
    trainval_list = []
    test_list = []
    trainval, test = walk_dir(devkit_dir)
    trainval_list.extend(trainval)
    test_list.extend(test)
    random.shuffle(trainval_list)
    with open(osp.join(output_dir, 'train.txt'), 'w') as ftrainval:
        for item in trainval_list:
            ftrainval.write(item[0] + ' ' + item[1] + '\n')

    with open(osp.join(output_dir, 'val.txt'), 'w') as ftest:
        for item in test_list:
            ftest.write(item[0] + ' ' + item[1] + '\n')


if __name__ == '__main__':
    prepare_filelist(devkit_dir, '.')



# import os
# import os.path as osp
# import re
# import random
# import locale

# devkit_dir = './'
# years = ['2007', '2012']

# def get_dir(devkit_dir,  type):
#     return osp.join(devkit_dir, type)

# def walk_dir(devkit_dir):
#     filelist_dir = get_dir(devkit_dir, 'ImageSets/Main')
#     annotation_dir = get_dir(devkit_dir, 'annotations')
#     img_dir = get_dir(devkit_dir, 'images')
#     trainval_list = []
#     test_list = []
#     added = set()

#     # Get the system's default encoding
#     system_encoding = locale.getpreferredencoding()

#     def read_file(fpath):
#         encodings = ['utf-8', system_encoding, 'latin1', 'utf-16']
#         for enc in encodings:
#             try:
#                 with open(fpath, encoding=enc) as f:
#                     return f.readlines()
#             except UnicodeDecodeError:
#                 continue
#         raise UnicodeDecodeError(f"Unable to decode {fpath} with tried encodings")

#     for _, _, files in os.walk(filelist_dir):
#         for fname in files:
#             img_ann_list = []
#             if re.match('train\.txt', fname):
#                 img_ann_list = trainval_list
#             elif re.match('val\.txt', fname):
#                 img_ann_list = test_list
#             else:
#                 continue
#             fpath = osp.join(filelist_dir, fname)
#             try:
#                 lines = read_file(fpath)
#                 for line in lines:
#                     name_prefix = line.strip().split()[0]
#                     if name_prefix in added:
#                         continue
#                     added.add(name_prefix)
#                     ann_path = osp.join(annotation_dir, name_prefix + '.xml')
#                     img_path = osp.join(img_dir, name_prefix + '.jpg')
#                     assert os.path.isfile(ann_path), f'file {ann_path} not found.'
#                     assert os.path.isfile(img_path), f'file {img_path} not found.'
#                     img_ann_list.append((img_path, ann_path))
#             except UnicodeDecodeError as e:
#                 print(f"Error decoding file {fpath}: {e}")
#             except AssertionError as e:
#                 print(e)

#     return trainval_list, test_list

# def prepare_filelist(devkit_dir, output_dir):
#     trainval_list = []
#     test_list = []
#     trainval, test = walk_dir(devkit_dir)
#     trainval_list.extend(trainval)
#     test_list.extend(test)
#     random.shuffle(trainval_list)
#     with open(osp.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as ftrainval:
#         for item in trainval_list:
#             ftrainval.write(item[0] + ' ' + item[1] + '\n')

#     with open(osp.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as ftest:
#         for item in test_list:
#             ftest.write(item[0] + ' ' + item[1] + '\n')

# if __name__ == '__main__':
#     prepare_filelist(devkit_dir, '.')
