import imageio
from PIL import Image
import os
path = './result'
Dirs = os.listdir(path)
print(Dirs)
for Dir in Dirs:
    Dir = '/'.join([path, Dir])
    file_names = sorted((fn for fn in os.listdir(Dir) if fn.endswith('.png')))
    print(file_names)
    images = list(map(lambda filename: imageio.imread('/'.join([Dir, filename])), file_names))
    imageio.mimsave('%s.gif' %Dir, images, duration = 0.2)
