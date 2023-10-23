import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader_palette
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import torch
import torchvision.utils as vutils

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.resize_or_crop = "none"

save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_loader = CreateDataLoader_palette(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)

    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx


for i, data in enumerate(dataset):
    print("Successfully process image %s" % str(i*opt.batchSize))
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst'] = data['inst'].half()
        data['image'] = data['image'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst'] = data['inst'].uint8()
        data['image'] = data['image'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)

    # temp = torch.randn_like(data['reference']) / 2
    # temp += data['reference']
    # temp = temp.clip(-1, 1)
    # data['reference'] = temp

    # temp = torch.randn_like(data['reference'])
    # temp = temp.clip(-1, 1)
    # data['reference'] = temp

    #
    # temp = torch.randn(1) / 10
    # temp = temp.clip(-1, 1)
    # data['reference'][0, 0, 0, 0] = temp


    minibatch = 1
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:
        generated = model.inference_palette(data['label'], data['image'], data['reference'])

    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']

    output_img = (generated + 1) / 2
    for j in range(opt.batchSize):
        try:
            out_img_name = data['path'][j].split("/")[-1].replace("jpg", "png")
            # out_img_palette_name = data['path'][j].split("/")[-1].replace("jpg", "png").replace(".png", "_palette.png")
            vutils.save_image(output_img[j:j+1], os.path.join(save_dir, out_img_name),
                              nrow=1, padding=0, normalize=False)
            # vutils.save_image(data['reference'], os.path.join(save_dir, out_img_palette_name),
            #                   nrow=1, padding=0, normalize=False)

        except OSError as err:
            print(err)
