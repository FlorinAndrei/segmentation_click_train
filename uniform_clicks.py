from simanneal import Annealer
import numpy as np
import pandas as pd
import pickle
import os
import sys
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import json


def compute_energy(points, region):
    """
    Compute the energy of each pixel in the image.
    The energy depends on the distance from the pixel to the points (clicks) assigned to the image.
    Shorter distance = higher energy.

    input:
    points = list of point (click) coordinates, one coordinate pair for each point (click)
    region = numpy array with the region we're trying to assign clicks to
    n_points = number of clicks
    """
    w, h = region.shape
    n_points = len(points)

    # Euclidian distance to each point on the map
    distances = np.zeros((w, h, n_points))
    # beware of the x/y order here!
    ycoords, xcoords = np.meshgrid(np.arange(w), np.arange(h))
    for i in range(n_points):
        p = points[i]
        cx, cy = (p[0], p[1])
        distances[:, :, i] = np.sqrt(((xcoords - cx) ** 2 + (ycoords - cy) ** 2))
    """
    # vectorized code appears to be actually slower than the explicit loop above
    # so it's commented out
    xcoords, ycoords, _ = np.meshgrid(np.arange(w), np.arange(h), np.arange(n_points))
    points_arr = np.array(points).T
    cx = np.broadcast_to(points_arr[0, :].reshape(1, 1, n_points), (w, h, n_points))
    cy = np.broadcast_to(points_arr[1, :].reshape(1, 1, n_points), (w, h, n_points))
    distances = np.sqrt((xcoords - cx) ** 2 + (ycoords - cy) ** 2)
    """
    # remove singularities
    distances[distances == 0.0] = 1.0

    # Energy as 1 / distance.
    # Closest point defines the pixel's energy (we only keep the max energy).
    # Points that are not closest to the pixel do not contribute energy.
    # This achieves an equivalent to Voronoi tesselation.
    # Energy is zero for pixels outside the region (this is why we multiply by region).
    energy = np.amax(np.dstack([region] * n_points) / distances, axis=2)
    # energy[energy < 0.0] = 0.0
    return energy


def generate_points(region, n_points, previous_state, time_left=1.0):
    # determine positions of points that belong to the region
    region_points = np.where(region == 1)

    bbox_col_diff = max(
        1, round((region_points[0].max() - region_points[0].min()) * time_left)
    )
    bbox_row_diff = max(
        1, round((region_points[1].max() - region_points[1].min()) * time_left)
    )
    if previous_state is not None:
        points = []
        for j in range(n_points):
            region_points_allowed = [[], []]
            for i in range(region_points[0].shape[0]):
                if (
                    region_points[0][i] <= previous_state[j][0] + bbox_col_diff
                    and region_points[0][i] >= previous_state[j][0] - bbox_col_diff
                    and region_points[1][i] <= previous_state[j][1] + bbox_row_diff
                    and region_points[1][i] >= previous_state[j][1] - bbox_row_diff
                ):
                    region_points_allowed[0].append(region_points[0][i])
                    region_points_allowed[1].append(region_points[1][i])
            region_points_allowed[0] = np.array(region_points_allowed[0])
            region_points_allowed[1] = np.array(region_points_allowed[1])
            region_points_allowed = tuple(region_points_allowed)

            # pick n_points randomly
            pick_indices = np.random.choice(region_points_allowed[0].shape[0], 1)
            # return point coordinates
            p_new = list(
                zip(
                    region_points_allowed[0][pick_indices],
                    region_points_allowed[1][pick_indices],
                )
            )
            points = points + p_new
    else:
        pick_indices = np.random.choice(region_points[0].shape[0], n_points)
        points = list(
            zip(
                region_points[0][pick_indices],
                region_points[1][pick_indices],
            )
        )
    return points


class SpreadPoints(Annealer):
    def __init__(
        self,
        baseline_dir,
        state,
        region,
        image_index=None,
        segment_index=None,
        segment_type=None,
    ):
        self.baseline_dir = baseline_dir
        self.region = region
        self.n_points = len(state)
        self.current_step = 0
        self.time_left = 0.0
        self.image_index = image_index
        self.segment_index = segment_index
        self.segment_type = segment_type
        super(SpreadPoints, self).__init__(state)

    def move(self):
        self.current_step += 1
        x = self.current_step / self.steps
        self.time_left = (1 - x) / (1 + x)
        self.state = generate_points(
            self.region, self.n_points, self.state, self.time_left
        )
        for p in self.state:
            assert self.region[p[0], p[1]] == 1

    def energy(self):
        energy = compute_energy(self.state, self.region)
        if self.time_left == 0.0:
            os.makedirs(
                self.baseline_dir
                + "/energy/"
                + str(self.image_index)
                + "/false_positives",
                exist_ok=True,
            )
            os.makedirs(
                self.baseline_dir
                + "/energy/"
                + str(self.image_index)
                + "/false_negatives",
                exist_ok=True,
            )
            os.makedirs(
                self.baseline_dir
                + "/energy/"
                + str(self.image_index)
                + "/true_positives",
                exist_ok=True,
            )
            with open(
                self.baseline_dir
                + "/energy/"
                + str(self.image_index)
                + "/"
                + self.segment_type
                + "/"
                + str(self.segment_index)
                + ".pkl",
                "wb",
            ) as energy_file:
                pickle.dump(energy, energy_file)

        return -energy.sum()


def test_gamma(worker_args):
    baseline_dir, frame_arr, sp_schedule, n_points = worker_args
    sp = SpreadPoints(
        baseline_dir=baseline_dir,
        state=generate_points(frame_arr, n_points, None),
        region=frame_arr,
        image_index=1,
        segment_index=1,
        segment_type="true_positives",
    )
    sp.set_schedule(sp_schedule)
    best_points, best_energy = sp.anneal()
    return best_points, best_energy


def click_generator(worker_args):
    (
        baseline_dir,
        image_index,
        segment_files,
        anneal_schedule,
        iou_max,
        max_clicks_per_segment,
        max_clicks_per_kind,
    ) = worker_args
    segment_stats = pd.read_csv(baseline_dir + "/segment_stats.csv", index_col=0)
    image_perf_df = pd.read_csv(baseline_dir + "/image_perf_df.csv", index_col=0)
    clicks = {}
    # keep true_positives at the top of the dict
    # true_positive must be computed first
    segment_kinds_and_click_names = {
        "true_positives": "clicks_positive_tp",
        "false_positives": "clicks_negative",
        "false_negatives": "clicks_positive",
    }

    for i in image_index:
        if i > image_perf_df.index.max():
            continue
        clicks[i] = {}
        for sk in segment_kinds_and_click_names.keys():
            clicks[i][segment_kinds_and_click_names[sk]] = []
        iou = image_perf_df.loc[i, "iou"]
        if iou > iou_max:
            continue

        image_stats = segment_stats[segment_stats["image_index"] == i].copy(deep=True)
        tp_total_area = 0
        for sk in segment_kinds_and_click_names.keys():
            click_name = segment_kinds_and_click_names[sk]
            clicks_per_kind = 0
            sk_stats_all = (
                image_stats[image_stats["segment_type"] == sk]
                .sort_values("segment_size", ascending=False)
                .copy(deep=True)
            )
            for s in sk_stats_all.iterrows():
                s_index = s[1]["segment_index"]
                s_size = s[1]["segment_size"]
                if sk == "true_positives":
                    tp_total_area += s_size

                # s_size < 1e3 px ==> n_points = 0
                # s_size = 1e3 px ==> n_points = 1
                # ... log scale ...
                # s_size = 1e5 px ==> n_points = 5
                # s_size > 1e5 px ==> n_points = 5
                n_points = min(
                    max_clicks_per_segment,
                    max(0, int(1 + (2 * (np.log10(s_size) - 3)))),
                )
                if n_points == 0:
                    if sk != "true_positives" and s_size >= 0.05 * tp_total_area:
                        # it's a small segment, but it's at least 5% of TP area
                        # so let's assign it one click
                        n_points = 1
                    else:
                        # assuming segments are already sorted by size
                        # otherwise this should be continue
                        break
                clicks_per_kind += n_points
                if clicks_per_kind > max_clicks_per_kind:
                    # reached maximum per kind of segments
                    break

                with open(
                    segment_files
                    + "/"
                    + str(i)
                    + "/"
                    + sk
                    + "/"
                    + str(s_index)
                    + ".pkl",
                    "rb",
                ) as s_file:
                    s_arr = pickle.load(s_file)
                sp = SpreadPoints(
                    baseline_dir=baseline_dir,
                    state=generate_points(s_arr, n_points, None),
                    region=s_arr,
                    image_index=i,
                    segment_index=s_index,
                    segment_type=sk,
                )
                sp.set_schedule(anneal_schedule)
                best_points, best_energy = sp.anneal()
                clicks[i][click_name] += [
                    [int(bp[0]), int(bp[1])] for bp in best_points
                ]
        if (
            len(clicks[i]["clicks_negative"]) == 0
            and len(clicks[i]["clicks_positive"]) == 0
        ):
            # if no corrections are needed for FP and FN,
            # then do not waste clicks on TP
            clicks[i]["clicks_positive_tp"] = []

    return clicks


if __name__ == "__main__":
    baseline_dir = sys.argv[1]
    guide_clicks_file = baseline_dir + "/guide_clicks.json"
    energy_dir = baseline_dir + "/energy"

    QUIT = False
    quit_message = ""
    if os.path.exists(energy_dir):
        QUIT = True
        quit_message += f"Segment energy maps directory {energy_dir} exists. "
    if os.path.exists(guide_clicks_file):
        QUIT = True
        quit_message += f"Guide clicks file {guide_clicks_file} exists. "
    if QUIT:
        quit_message += "Exit."
        print(quit_message)
        exit()

    # images with IoU higher than or equal to this do not get clicks
    iou_max = 0.95
    # hard limit of number of clicks per contiguous area (safety limit)
    max_clicks_per_segment = 5
    # maximum count of either positive clicks, or negative clicks, per image
    max_clicks_per_kind = max_clicks_per_segment * 2

    anneal_schedule = {"tmax": 2000.0, "tmin": 0.1, "steps": 100, "updates": 0}

    # energy_dir = baseline_dir + "/energy"
    # shutil.rmtree(energy_dir, ignore_errors=True)
    # os.makedirs(energy_dir, exist_ok=True)

    image_perf_df = pd.read_csv(baseline_dir + "/image_perf_df.csv")

    segment_files = baseline_dir + "/segment_files"

    n_cpus = multiprocessing.cpu_count()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    p = Pool(processes=n_cpus)
    mp_results = []

    step_multiplier = 10
    step_size = n_cpus * step_multiplier

    for i in tqdm(range(0, image_perf_df.shape[0], step_size)):
        rows = image_perf_df.loc[i : i + step_size - 1, :]
        rows_index = rows.index.to_list()
        arglist = list(
            zip(
                [baseline_dir] * n_cpus,
                [
                    list(
                        range(
                            i + step_multiplier * j,
                            i + step_multiplier * j + step_multiplier,
                        )
                    )
                    for j in range(n_cpus)
                ],
                [segment_files] * n_cpus,
                [anneal_schedule] * n_cpus,
                [iou_max] * n_cpus,
                [max_clicks_per_segment] * n_cpus,
                [max_clicks_per_kind] * n_cpus,
            )
        )
        mp_results_step = p.map(click_generator, arglist)
        mp_results += mp_results_step

    p.close()

    guide_clicks = {}
    for res in mp_results:
        for k, v in res.items():
            guide_clicks[k] = v

    with open(guide_clicks_file, "w") as gcf:
        json.dump(guide_clicks, gcf, indent=2)
