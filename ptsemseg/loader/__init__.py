import json

from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.mit_sceneparsing_benchmark_loader import (
    MITSceneParsingBenchmarkLoader
)
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.nyuv2_loader import NYUv2Loader
from ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
from ptsemseg.loader.mapillary_vistas_loader import mapillaryVistasLoader

from ptsemseg.loader.my_loader import myLoader
from ptsemseg.loader.HRS_sar_seg_loader import HRS_SAR_seg_Loader
from ptsemseg.loader.HRS_sar_seg_loader_chen import HRS_SAR_seg_chen_Loader
from ptsemseg.loader.SAR_CD_loader import  *
from ptsemseg.loader.PolSAR_CD2 import *
import yaml
def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "my":myLoader,
        "HRS_SAR_seg": HRS_SAR_seg_Loader,
        "HRS_SAR_seg_chen":HRS_SAR_seg_chen_Loader,
        'SAR_CD_direct':SAR_CD_direct,
        'SAR_CD_intensity':SAR_CD_intensities,
        'SAR_CD_Hoekman':SAR_CD_Hoekman,
        'SAR_CD_tile_1': SAR_CD_tile_1,
        'PolSAR_CD_base': PolSAR_CD_base,
    }[name]