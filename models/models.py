import torch

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == "reference_model":
        from .reference_model import ReferenceModel, InferenceModel
        if opt.isTrain:
            model = ReferenceModel()
        else:
            model = InferenceModel()
    elif opt.model == "reference_model_palette":
        from .reference_model_palette import ReferenceModel, InferenceModel
        if opt.isTrain:
            model = ReferenceModel()
        else:
            model = InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
