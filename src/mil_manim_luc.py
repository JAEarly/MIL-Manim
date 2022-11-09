import matplotlib as mpl
import numpy as np
from manim import *

from util import ShrinkToPoint, ArrayMobject, create_filter, calculate_angle


class MILManimLUC(Scene):

    def construct(self):
        # Setup scene
        self.camera.background_color = WHITE
        cmap = mpl.cm.get_cmap('viridis')

        # Setup grid of patches (no animation)
        grid_size = (3, 3)
        n_patches = grid_size[0] * grid_size[1]
        patches = np.empty(grid_size, dtype=object)
        for col in range(grid_size[0]):
            for row in range(grid_size[1]):
                patch_path = "img/lcc/dg_lcc_{:d}_{:d}.png".format(row, col)
                patch = ImageMobject(patch_path)
                patch.height = patch.width = 1
                patch.shift((col - 1) * RIGHT + (row - 1) * DOWN).set_z_index(1)
                patches[row][col] = patch
        flat_patches = patches.ravel()

        # Setup random feature vectors
        n_features = 7
        np.random.seed(0)
        feature_vectors = np.random.rand(n_patches, n_features) * 2 - 1
        features = []
        for idx in range(n_patches):
            features.append(ArrayMobject(feature_vectors[idx], cmap, 0, 1).create_mobject().set_z_index(1))

        # Setup instance and bag predictions
        #   urban_land, agriculture_land, rangeland, forest_land, water, barren_land, unknown
        instance_preds = [
            [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.7, 0.1, 0.0, 0.0, 0.1, 0.0],
            [0.0, 0.8, 0.1, 0.0, 0.0, 0.1, 0.0],
            [0.2, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.6, 0.2, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.1, 0.6, 0.0, 0.0, 0.0],
        ]
        instance_preds = np.asarray(instance_preds)
        instance_pred_objs = []
        for idx in range(n_patches):
            instance_pred_objs.append(ArrayMobject(instance_preds[idx], cmap, -1, 1).create_mobject().set_z_index(1))
        bag_pred = np.mean(instance_preds, axis=0)
        bag_pred_obj = ArrayMobject(bag_pred, cmap, 0, 1)
        bag_pred_obj = bag_pred_obj.create_mobject().set_z_index(1)

        # Intro text
        intro_text_1 = Text("Multiple Instance Learning", font_size=50, color=BLACK).shift(UP)
        intro_text_2 = Text("Land Cover Classification", font_size=50, color=BLACK)
        intro_text_3 = Text("Scene-to-Patch Model Pipeline", font_size=50, color=BLACK).shift(DOWN)
        self.play(Write(intro_text_1))
        self.play(Write(intro_text_2))
        self.play(Write(intro_text_3))
        self.wait(2)
        self.play(
            Unwrite(intro_text_1),
            Unwrite(intro_text_2),
            Unwrite(intro_text_3),
        )
        self.wait(1)

        # Add patches to scene to look like one image
        orig_img_text = Text("Scene Image", font_size=50, color=BLACK).shift(UP * 2.5)
        self.play(Write(orig_img_text))
        self.play(*[FadeIn(patch) for patch in flat_patches])
        self.wait(1)
        orig_img_label_text = Text("Urban: 14% \n"
                                   "Agricultural: 65% \n"
                                   "Rangeland: 14% \n"
                                   "Forest: 5%\n"
                                   "Water: 0% \n"
                                   "Barren: 2%\n"
                                   "Unknown: 0%",
                                   font_size=25, color=BLACK, slant=ITALIC).shift(RIGHT * 3.2)
        self.play(Write(orig_img_label_text), run_time=3)
        self.wait(5)
        self.play(
            Unwrite(orig_img_text),
            Unwrite(orig_img_label_text),
        )

        # Create and add grid lines to divide patches
        lines = []
        for col in range(grid_size[0] - 1):
            line = Line((col - 0.5) * RIGHT - 1.55 * DOWN, (col - 0.5) * RIGHT + 1.55 * DOWN).set_z_index(2)
            lines.append(line)
        for row in range(grid_size[1] - 1):
            line = Line(-1.55 * RIGHT + (row - 0.5) * DOWN, 1.55 * RIGHT + (row - 0.5) * DOWN).set_z_index(2)
            lines.append(line)
        self.play(*[Create(line) for line in lines])
        self.wait(1)

        # Create and play animations to split image into patches
        patches_text = Text("Patches", font_size=50, color=BLACK).shift(UP * 2.5)
        split_anims = [Write(patches_text)]
        for col in range(grid_size[0]):
            for row in range(grid_size[1]):
                square = patches[row][col]
                new_loc = (0.2 * (col - 1) * RIGHT + 0.2 * (row - 1) * DOWN)
                split_anims.append(square.animate.shift(new_loc))
        for line in lines:
            split_anims.append(FadeOut(line))
        self.play(*split_anims)
        self.remove(*lines)
        self.wait(2)
        self.play(Unwrite(patches_text))

        # Create and play animations to align patches in a column
        flatten_anims = []
        for col in range(grid_size[0]):
            for row in range(grid_size[1]):
                patch = patches[row][col]
                new_loc = 2.7 * UP + 6 * LEFT + (col * 0.7 + 2.1 * row) * DOWN
                patch.generate_target()
                patch.target.scale(0.6)
                patch.target.move_to(new_loc)
                flatten_anims.append(MoveToTarget(patch))
        self.play(*flatten_anims)
        self.wait(1)
        bag_text = Text("Bag", font_size=50, color=BLACK).shift(UP * 3.5 + 6 * LEFT)
        self.play(Write(bag_text))
        self.wait(1)

        # Create and add feature extractor
        fe_filter = create_filter(GREEN).scale(0.4)
        fe_filter.set_z_index(2).move_to(flat_patches[0].get_center() + 1.5 * RIGHT).rotate(PI/2)
        fe_text = Text("   Feature   \nExtractor", font_size=20, color=BLACK).shift(UP * 3.6 + 4.5 * LEFT)
        self.play(
            Write(fe_text),
            Create(fe_filter),
        )
        self.wait(1)

        # Convert patches into features
        feature_text = Text("Features", font_size=50, color=BLACK).shift(UP * 3.6 + 2.3 * LEFT)
        patch_copies = [p.copy().set_z_index(0) for p in flat_patches]
        for idx, patch in enumerate(flat_patches):
            patch_copy = patch_copies[idx]
            patch_copy.generate_target()
            patch_copy.target.fade(0.5)
            run_time = 1 if idx < 3 else 0.3
            self.play(
                ShrinkToPoint(patch, fe_filter.get_center()),
                MoveToTarget(patch_copy),
                run_time=run_time
            )
            features[idx].move_to(patch.get_center() + 2.2 * RIGHT).scale(0.4)
            self.play(
                Indicate(fe_filter),
                GrowFromPoint(features[idx], fe_filter.get_center() + 0.5 * RIGHT),
                run_time=run_time
            )
            if idx == 0:
                self.play(Write(feature_text))
            if idx < len(flat_patches) - 1:
                self.play(fe_filter.animate.move_to(flat_patches[idx + 1].get_center() + 1.5 * RIGHT),
                          run_time=run_time)
            self.remove(patch)
        self.wait(1)

        # Shift features left
        self.play(
            Uncreate(fe_filter),
            Unwrite(fe_text),
        )
        feature_text.generate_target()
        feature_text.target.next_to(bag_text, buff=0.5).shift(UP * 0.07)
        self.play(
            MoveToTarget(feature_text),
            *[f.animate.move_to([feature_text.target.get_x(), f.get_y(), 0]) for f in features],
        )
        self.wait(1)

        # Create and add classifier
        clz_filter = create_filter(RED).scale(0.4)
        clz_filter.set_z_index(2).move_to(features[0].get_center() + 2.3 * RIGHT).rotate(PI/2)
        clz_text = Text("  Instance  \nClassifier", font_size=20, color=BLACK).shift(UP * 3.6 + 1.3 * LEFT)

        self.play(
            Write(clz_text),
            Create(clz_filter),
        )
        self.wait(1)

        # Classify patches
        feature_copies = [f.copy().set_z_index(0) for f in features]
        instance_preds_text = Text("  Instance  \nPredictions",
                                   font_size=28, color=BLACK).next_to(clz_text, buff=0.6).shift(DOWN * 0.1)
        for idx, feature in enumerate(features):
            feature_copy = feature_copies[idx]
            feature_copy.generate_target()
            feature_copy.target.fade(0.5)
            run_time = 1 if idx < 3 else 0.3
            self.play(
                ShrinkToPoint(feature, clz_filter.get_center()),
                MoveToTarget(feature_copy),
                run_time=run_time
            )
            instance_pred_objs[idx].move_to(feature.get_center() + 2.2 * RIGHT).scale(0.4)
            self.play(
                Indicate(fe_filter),
                GrowFromPoint(instance_pred_objs[idx], clz_filter.get_center() + 0.5 * RIGHT),
                run_time=run_time
            )
            if idx == 0:
                self.play(Write(instance_preds_text))
            if idx < len(features) - 1:
                self.play(clz_filter.animate.move_to(features[idx + 1].get_center() + 2.3 * RIGHT),
                          run_time=run_time)
            self.remove(feature)
        self.wait(1)

        # Shift instance predictions left
        self.play(
            Uncreate(clz_filter),
            Unwrite(clz_text),
        )
        instance_preds_text.generate_target()
        instance_preds_text.target.next_to(feature_text, buff=1.2).shift(DOWN * 0.07)
        self.play(
            MoveToTarget(instance_preds_text),
            *[o.animate.move_to([instance_preds_text.target.get_x(), o.get_y(), 0]) for o in instance_pred_objs],
        )
        self.wait(1)

        # Merge instance predictions into bag prediction
        instance_pred_obj_copies = [o.copy().set_z_index(0) for o in instance_pred_objs]
        bag_pred_text = Text("     Bag     \nPrediction",
                             font_size=28, color=BLACK).next_to(instance_preds_text, buff=2.2)#.shift(DOWN * 0.1)
        bag_pred_obj.move_to(instance_pred_objs[0].get_center() + 4 * RIGHT).scale(0.4)
        for idx in range(len(instance_pred_objs)):
            instance_pred_obj_copies[idx].generate_target()
            instance_pred_obj_copies[idx].target.fade(0.5)
        self.play(
            *[ShrinkToPoint(o, bag_pred_obj.get_center()) for o in instance_pred_objs],
            *[MoveToTarget(c) for c in instance_pred_obj_copies],
            GrowFromCenter(bag_pred_obj),
            Write(bag_pred_text),
            run_time=3
        )

        # Write bag clz predictions
        clz_bag_pred_text = Text("Urban: 21% \n"
                                 "Agricultural: 57% \n"
                                 "Rangeland: 12% \n"
                                 "Forest: 8%\n"
                                 "Water: 0% \n"
                                 "Barren: 2%\n"
                                 "Unknown: 0%",
                                 font_size=25, color=BLACK, slant=ITALIC)
        clz_bag_pred_text.move_to(bag_pred_obj.get_center() + 1.5 * DOWN)
        self.play(Write(clz_bag_pred_text), run_time=3)
        self.wait(3)

        # Create instance prediction grid (with overlay?)
        grid_size = (3, 3)
        final_patches = np.empty(grid_size, dtype=object)
        final_patch_size = 0.7
        final_patch_colours = [YELLOW, YELLOW, YELLOW, YELLOW, BLUE, YELLOW, YELLOW, BLUE, GREEN]
        for col in range(grid_size[0]):
            for row in range(grid_size[1]):
                patch_path = "img/lcc/dg_lcc_{:d}_{:d}.png".format(row, col)
                final_patch = ImageMobject(patch_path)
                final_patch.height = final_patch.width = final_patch_size
                final_patch.shift((2.5 + col * final_patch_size) * RIGHT
                                  + (1.5 + row * final_patch_size) * DOWN).set_z_index(0)
                final_patches[row][col] = final_patch
        final_flat_patches = final_patches.ravel()
        patch_pred_text = Text("Patch Prediction Mask", font_size=28, color=BLACK)
        patch_pred_text.next_to(final_patches[1][1], direction=UP, buff=1).shift(1.1 * RIGHT)
        self.play(
            *[FadeIn(p) for p in final_flat_patches],
            Write(patch_pred_text),
        )
        for idx, final_patch in enumerate(final_flat_patches):
            square = Square()
            square.move_to(final_patch)\
                .set_z_index(1)\
                .set_fill(color=final_patch_colours[idx], opacity=0.5)\
                .set_stroke(width=0)
            square.height = square.width = final_patch_size
            self.play(GrowFromPoint(square, instance_pred_obj_copies[idx].get_center()))

        agricultural_text = Text("Agricultural", font_size=28, color=YELLOW_E).next_to(final_patches[0][2], buff=0.3)
        urban_text = Text("Urban", font_size=28, color=BLUE).next_to(final_patches[1][2], buff=0.3)
        forest_text = Text("Forest", font_size=28, color=GREEN).next_to(final_patches[2][2], buff=0.3)
        self.wait(1)
        self.play(
            Write(agricultural_text),
            Write(urban_text),
            Write(forest_text),
            run_time=2
        )
        self.wait(1)

        # Fade all out
        # self.play(
        #     *[FadeOut(p) for p in patch_copies],
        #     *[FadeOut(f) for f in feature_copies],
        #     FadeOut(agg_fv_copy),
        #     FadeOut(pred_0_text),
        #     FadeOut(pred_1_text),
        #     FadeOut(splits[0]),
        #     FadeOut(splits[1]),
        #     FadeOut(bag_text),
        #     FadeOut(feature_text),
        #     FadeOut(agg_text),
        #     FadeOut(pred_text),
        # )
        # self.wait(1)
