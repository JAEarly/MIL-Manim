import matplotlib as mpl
import torch
from manim import *

from util import ShrinkToPoint, ArrayMobject, create_filter, calculate_angle


class MILManim(Scene):

    def construct(self):
        # Setup scene
        self.camera.background_color = WHITE
        cmap = mpl.cm.get_cmap('viridis')

        # Intro text
        intro_text_1 = Text("Multiple Instance Learning", font_size=50, color=BLACK).shift(UP)
        intro_text_2 = Text("Model Pipeline", font_size=50, color=BLACK).shift(DOWN)
        self.play(Write(intro_text_1))
        self.play(Write(intro_text_2))
        self.wait(2)
        self.play(
            Unwrite(intro_text_1),
            Unwrite(intro_text_2),
        )
        self.wait(1)

        # Create grid of patches
        grid_size = (3, 3)
        n_patches = grid_size[0] * grid_size[1]
        patches = np.empty(grid_size, dtype=object)
        for col in range(grid_size[0]):
            for row in range(grid_size[1]):
                patch_path = "img/crc_{:d}_{:d}.png".format(row, col)
                patch = ImageMobject(patch_path)
                patch.height = patch.width = 1
                patch.shift((col - 1) * RIGHT + (row - 1) * DOWN).set_z_index(1)
                # self.play(Create(square))
                patches[row][col] = patch
        flat_patches = patches.ravel()

        # Create random extracted features
        n_features = 7
        torch.random.manual_seed(0)
        feature_vectors = torch.rand((n_patches, n_features)) * 2 - 1
        agg_fv = torch.rand((n_features, 1)) * 2 - 1
        features = []
        for idx in range(n_patches):
            features.append(ArrayMobject(feature_vectors[idx], cmap, -1, 1).create_mobject().set_z_index(1))
        agg_fv = ArrayMobject(agg_fv, cmap, -1, 1).create_mobject().set_z_index(1)
        pred_obj = ArrayMobject(torch.as_tensor([0.1, 0.9]), cmap, 0, 1)
        pred_fv = pred_obj.create_mobject().set_z_index(1)

        # Add patches to scene to look like one image
        orig_img_text = Text("Original Image", font_size=50, color=BLACK).shift(UP * 2.5)
        self.play(Write(orig_img_text))
        self.play(*[FadeIn(patch) for patch in flat_patches])
        self.wait(1)
        self.play(Unwrite(orig_img_text))

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
        fe_text = Text("Feature \nExtractor", font_size=20, color=BLACK).shift(UP * 3.6 + 4.5 * LEFT)
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

        # Create and add aggregator
        agg_filter = create_filter(BLUE).scale(0.4)
        agg_filter.set_z_index(2).move_to(features[4].get_center() + 2.3 * RIGHT).rotate(PI/2)
        aggregator_text = Text("Aggregator", font_size=20, color=BLACK).next_to(feature_text, buff=0.2).shift(DOWN * 0.1)
        self.play(
            Write(aggregator_text),
            Create(agg_filter),
        )
        self.wait(1)

        # Aggregate features
        #  Here we track the current rotation of the aggregator and rotate it by the difference between its current
        #   rotation and the new target rotation, as Manim rotation is relative rather than absolute
        feature_copies = [f.copy().set_z_index(0) for f in features]
        current_rotation = calculate_angle(agg_filter, features[0]) - PI
        self.play(agg_filter.animate.rotate(current_rotation, about_point=agg_filter.get_center_of_mass()))
        for idx, feature in enumerate(features):
            run_time = 1 if idx < 3 else 0.3
            feature_copy = feature_copies[idx]
            feature_copy.generate_target()
            feature_copy.target.fade(0.5)
            self.play(
                ShrinkToPoint(feature, agg_filter.get_center()),
                MoveToTarget(feature_copy),
                run_time=run_time,
            )
            if idx < len(flat_patches) - 1:
                # Quick fix to alignment of middle patch as it doesn't quick align properly normally (rounding error?)
                new_rotation = 0 if idx == 3 else calculate_angle(agg_filter, features[idx + 1]) - PI
                self.play(
                    agg_filter.animate.rotate(new_rotation - current_rotation,
                                              about_point=agg_filter.get_center_of_mass()),
                    run_time=run_time,
                )
                current_rotation = new_rotation
            self.remove(feature)
        self.wait(1)
        self.play(agg_filter.animate.rotate(-current_rotation,
                                            about_point=agg_filter.get_center_of_mass()))
        self.wait(1)

        # Create bag aggregation
        agg_text = Text("Aggregation", font_size=50, color=BLACK).next_to(aggregator_text, buff=0.2).shift(UP * 0.06)
        agg_fv.move_to([agg_text.get_x(), agg_filter.get_y(), 0]).scale(0.5)
        self.play(
            Indicate(agg_filter),
            GrowFromPoint(agg_fv, agg_filter.get_center() + 0.5 * RIGHT),
            Write(agg_text),
        )
        self.wait(1)

        # Shift aggregation left
        self.play(
            Uncreate(agg_filter),
            Unwrite(aggregator_text),
        )
        agg_text.generate_target()
        agg_text.target.next_to(feature_text, buff=0.5).shift(DOWN * 0.05)
        self.play(
            MoveToTarget(agg_text),
            agg_fv.animate.move_to([agg_text.target.get_x(), agg_fv.get_y(), 0]),
        )
        self.wait(1)

        # Create classifier
        clz_filter = create_filter(RED).scale(0.4)
        clz_filter.set_z_index(2).next_to(agg_fv, buff=0.5).rotate(PI / 2)
        clz_text = Text("Classifier", font_size=20, color=BLACK).next_to(agg_text, buff=0.2)
        self.play(
            Write(clz_text),
            Create(clz_filter),
        )
        self.wait(1)

        # Run classification
        agg_fv_copy = agg_fv.copy().set_z_index(0)
        agg_fv_copy.generate_target()
        agg_fv_copy.target.fade(0.5)
        self.play(
            ShrinkToPoint(agg_fv, clz_filter.get_center()),
            MoveToTarget(agg_fv_copy),
        )
        self.wait(1)
        pred_text = Text("Prediction", font_size=50, color=BLACK).next_to(clz_text, buff=0.2).shift(UP * 0.06)
        pred_fv.move_to([pred_text.get_x(), clz_filter.get_y(), 0]).scale(0.7)
        self.play(
            Indicate(clz_filter),
            GrowFromPoint(pred_fv, clz_filter.get_center() + 0.5 * RIGHT),
            Write(pred_text),
        )
        self.wait(1)

        # Shift aggregation left
        self.play(
            Uncreate(clz_filter),
            Unwrite(clz_text),
        )
        pred_text.generate_target()
        pred_text.target.next_to(agg_text, buff=0.5).shift(UP * 0.05)
        self.play(
            MoveToTarget(pred_text),
            pred_fv.animate.move_to([pred_text.target.get_x(), pred_fv.get_y(), 0]),
        )
        self.wait(1)

        # Show final outputs
        splits = pred_obj.create_splits()
        splits[0].move_to(pred_fv.get_center() + pred_fv.width/4 * LEFT).scale(0.7).set_z_index(2)
        splits[1].move_to(pred_fv.get_center() + pred_fv.width/4 * RIGHT).scale(0.7).set_z_index(2)
        self.add(splits[0])
        self.add(splits[1])
        self.remove(pred_fv)
        splits[0].generate_target()
        splits[1].generate_target()
        splits[0].target.move_to(DOWN + 6 * RIGHT)
        splits[1].target.move_to(UP + 6 * RIGHT)
        pred_0_text = Text("Non-Epithelial:", font_size=30, color=BLACK).next_to(splits[0].target, direction=LEFT,
                                                                                 buff=0.2)
        pred_1_text = Text("Epithelial:", font_size=30, color=BLACK).next_to(splits[1].target, direction=LEFT,
                                                                             buff=0.2)
        self.play(
            MoveToTarget(splits[0]),
            MoveToTarget(splits[1]),
        )
        self.play(
            Write(pred_0_text),
            Write(pred_1_text),
        )
        self.wait(2)

        # Fade all out
        self.play(
            *[FadeOut(p) for p in patch_copies],
            *[FadeOut(f) for f in feature_copies],
            FadeOut(agg_fv_copy),
            FadeOut(pred_0_text),
            FadeOut(pred_1_text),
            FadeOut(splits[0]),
            FadeOut(splits[1]),
            FadeOut(bag_text),
            FadeOut(feature_text),
            FadeOut(agg_text),
            FadeOut(pred_text),
        )
        self.wait(1)
