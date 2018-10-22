import logging
import PIL.Image
import io
import numpy as np
import utils.interpolations

LOG = logging.getLogger(__name__)


def log(func):
    def decorator(*args, **kwargs):
        LOG.info('Running %s...' % func.__name__)
        return func(*args, **kwargs)

    return decorator

@log
def init_hook(**params):
    LOG.info("Init hooks {}".format(params))



@log
def preprocess(inputs,**kwargs):
    LOG.info('Preprocess: {}, args: {}'.format(inputs,kwargs))
    images = inputs['image']
    image = PIL.Image.open(io.BytesIO(images[0]))
    image = image.resize((64,64))
    image = np.asarray(image)
    image = image / 127.5 - 1
    image = np.transpose(image, (2,0,1))
    image = np.reshape(image, (1,image.shape[0],image.shape[1], image.shape[2]))
    image = np.repeat(image, 32, axis=0)
    z_vectors = utils.interpolations.create_mine_grid(rows=64, cols=64,
                                                      dim=100, space=3, anchors=None,
                                                      spherical=True, gaussian=True)
    return {'i': image,'z': z_vectors[0:32]}

@log
def postprocess(outputs,**kwargs):
    for k,v in outputs.items():
        outputs = v[0]
        LOG.info('Use {} as output,{}'.format(k,v.shape))
        break
    outputs = np.transpose(outputs, (1,2,0))
    outputs = (outputs + 1) / 2 * 255
    outputs = np.uint8(np.clip(outputs, 0, 255.0))
    h = []
    images = []
    for i in range(outputs.shape[0]):
        h.append(outputs[i])
        if len(h) == 8:
            images.append(np.concatenate(h, axis=1))
            h = []
        if len(images) == 4:
            break
    images = np.concatenate(images, axis=0)
    im = PIL.Image.fromarray(images)
    with io.BytesIO() as output:
        im.save(output,format='PNG')
        contents = output.getvalue()
    return {'output':contents}