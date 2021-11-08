import itertools
import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.ovis import OVIS
from pycocotools.oviseval import OVISeval
from terminaltables import AsciiTable
from .recall import eval_recalls


def ovis_eval_vis(result_file, result_types, ovis_dataset, max_dets=(100, 300, 1000)):

    ovis = ovis_dataset if isinstance(ovis_dataset, str) else ovis_dataset.ovis
    if mmcv.is_str(ovis):
        ovis = OVIS(ovis)
    assert isinstance(ovis, OVIS)

    if len(ovis.anns) == 0:
        print("Annotations does not exist")
        return
    assert result_file.endswith('.json')
    ovis_dets = ovis.loadRes(result_file)

    vid_ids = ovis.getVidIds()
    for res_type in result_types:
        iou_type = res_type
        ovisEval = OVISeval(ovis, ovis_dets, iou_type)
        if res_type == 'proposal':
            ovisEval.params.useCats = 0
            ovisEval.params.maxDets = list(max_dets)
        ovisEval.evaluate()
        ovisEval.accumulate()
        ovisEval.summarize()

        # Compute per-category AP
        # from https://github.com/facebookresearch/detectron2/
        precisions = ovisEval.eval['precision']
        # precision: (iou, recall, cls, area range, max dets)
        assert len(ovis_dataset.cat_ids) == precisions.shape[2]

        results_per_category = []
        for idx, catId in enumerate(ovis_dataset.cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = ovis_dataset.ovis.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (f'{nm["name"]}', f'{float(ap):0.3f}'))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(
            itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(*[
            results_flatten[i::num_columns]
            for i in range(num_columns)
        ])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print('\n' + table.table)



def ovis_eval(result_file, result_types, ovis, max_dets=(100, 300, 1000)):

    if mmcv.is_str(ovis):
        ovis = OVIS(ovis)
    assert isinstance(ovis, OVIS)

    if len(ovis.anns) == 0:
        print("Annotations does not exist")
        return
    assert result_file.endswith('.json')
    ovis_dets = ovis.loadRes(result_file)

    vid_ids = ovis.getVidIds()
    for res_type in result_types:
        iou_type = res_type
        ovisEval = OVISeval(ovis, ovis_dets, iou_type)
        ovisEval.params.vidIds = vid_ids
        if res_type == 'proposal':
            ovisEval.params.useCats = 0
            ovisEval.params.maxDets = list(max_dets)
        ovisEval.evaluate()
        ovisEval.accumulate()
        ovisEval.summarize()


def coco_eval(result_file, result_types, coco, max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_file, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    assert result_file.endswith('.json')
    coco_dets = coco.loadRes(result_file)

    img_ids = coco.getImgIds()
    for res_type in result_types:
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


def fast_eval_recall(results,
                     coco,
                     max_dets,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
            format(type(results)))

    gt_bboxes = []
    img_ids = coco.getImgIds()
    for i in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=False)
    ar = recalls.mean(axis=1)
    return ar


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg = results[idx]
        for label in range(len(det)):
            bboxes = det[label]
            segms = seg[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                json_results.append(data)
    return json_results


def results2json_videoseg(dataset, results, out_file):
    json_results = []
    vid_objs = {}
    for idx in range(len(dataset)):
      # assume results is ordered

      vid_id, frame_id = dataset.img_ids[idx]
      if idx == len(dataset) - 1 :
        is_last = True
      else:
        _, frame_id_next = dataset.img_ids[idx+1]
        is_last = frame_id_next == 0
      det, seg = results[idx]
      for obj_id in det:
        bbox = det[obj_id]['bbox']
        segm = seg[obj_id]
        label = det[obj_id]['label']
        if obj_id not in vid_objs:
          vid_objs[obj_id] = {'scores':[], 'cats':[], 'cats_perframe':{}, 'segms':{}, 'bboxes':{}}
        vid_objs[obj_id]['scores'].append(bbox[4])
        vid_objs[obj_id]['cats'].append(label)
        vid_objs[obj_id]['cats_perframe'][frame_id] = label
        segm['counts'] = segm['counts'].decode()
        vid_objs[obj_id]['segms'][frame_id] = segm
        vid_objs[obj_id]['bboxes'][frame_id] = bbox[:4]
      if is_last:
        # store results of  the current video
        for obj_id, obj in vid_objs.items():
          data = dict()

          data['video_id'] = vid_id + 1
          data['score'] = np.array(obj['scores']).mean().item()
          # majority voting for sequence category
          data['category_id'] = np.bincount(np.array(obj['cats'])).argmax().item() + 1
          vid_seg = []
          vid_box = []
          vid_cat = []
          for fid in range(frame_id + 1):
            if fid in obj['segms']:
              vid_seg.append(obj['segms'][fid])
              vid_box.append(obj['bboxes'][fid].tolist())
              vid_cat.append(obj['cats_perframe'][fid].item())
            else:
              vid_seg.append(None)
              vid_box.append(None)
              vid_cat.append(None)
          data['segmentations'] = vid_seg
          data['detections'] = vid_box
          data['cats'] = vid_cat
          json_results.append(data)
        vid_objs = {}
    mmcv.dump(json_results, out_file)

def results2json(dataset, results, out_file):
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
    else:
        raise TypeError('invalid type of results')
    mmcv.dump(json_results, out_file)
