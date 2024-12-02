import numpy as np
from bokeh.models import (
    CheckboxGroup,
    ColumnDataSource,
    CustomJSHover,
    Ellipse,
    HoverTool,
    Scatter,
    Text,
)

js_resolution = """
    var detector_distance = params.data.detector_distance
    var beam_energy = params.data.beam_energy
    var beam_center_x = params.data.beam_center_x
    var beam_center_y = params.data.beam_center_y

    var x = special_vars.x - beam_center_x
    var y = special_vars.y - beam_center_y

    var theta = Math.atan(Math.sqrt(x*x + y*y) * 75e-6 / detector_distance) / 2
    var resolution = 6200 / beam_energy / Math.sin(theta)  // 6200 = 1.24 / 2 / 1e-4

    return resolution.toFixed(2)
"""

POSITIONS = (1, 1.2, 1.4, 1.5, 1.6, 1.8, 2, 2.2, 2.6, 3, 5, 10)


class ResolutionRings:
    def __init__(self, image_views, sv_metadata, sv_streamctrl, positions=POSITIONS):
        """Initialize a resolution rings overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
            sv_metadata (MetadataHandler): A metadata handler to report metadata issues.
            sv_streamctrl (StreamControl): A StreamControl instance of an application.
            positions (list, optional): Scattering radii in Angstroms. Defaults to
                [1.4, 1.5, 1.6, 1.8, 2, 2.2, 2.6, 3, 5, 10].
        """
        self._sv_metadata = sv_metadata
        self._sv_streamctrl = sv_streamctrl
        self.positions = np.array(positions)

        # ---- add resolution tooltip to hover tool
        self._formatter_source = ColumnDataSource(
            data=dict(
                detector_distance=[np.nan],
                beam_energy=[np.nan],
                beam_center_x=[np.nan],
                beam_center_y=[np.nan],
            )
        )

        resolution_formatter = CustomJSHover(
            args=dict(params=self._formatter_source), code=js_resolution
        )

        # ---- resolution rings
        self._source = ColumnDataSource(dict(x=[], y=[], w=[], h=[], text_x=[], text_y=[], text=[]))
        ellipse_glyph = Ellipse(
            x="x", y="y", width="w", height="h", fill_alpha=0, line_color="white"
        )

        text_glyph = Text(
            x="text_x",
            y="text_y",
            text="text",
            text_align="center",
            text_baseline="middle",
            text_color="white",
        )

        cross_glyph = Scatter(
            x="beam_center_x", y="beam_center_y", marker="cross", size=15, line_color="red"
        )

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, ellipse_glyph)
            image_view.plot.add_glyph(self._source, text_glyph)
            image_view.plot.add_glyph(self._formatter_source, cross_glyph)

        # ---- switch
        def switch_callback(_attr, _old, new):
            for image_view in image_views:
                image_renderer = image_view.plot.select(name="image_glyph")
                if 0 in new:
                    hovertool = HoverTool(
                        tooltips=[("intensity", "@image"), ("resolution", "@x{resolution} Å")],
                        formatters={"@x": resolution_formatter},
                        renderers=image_renderer,
                    )
                else:
                    hovertool = HoverTool(
                        tooltips=[("intensity", "@image")], renderers=image_renderer
                    )

                image_view.plot.tools[-1] = hovertool

        switch = CheckboxGroup(labels=["Resolution Rings"], width=145, margin=(0, 5, 0, 5))
        switch.on_change("active", switch_callback)
        self.switch = switch

    def _clear(self):
        if len(self._source.data["x"]):
            self._source.data.update(x=[], y=[], w=[], h=[], text_x=[], text_y=[], text=[])

    def update(self, metadata):
        """Trigger an update for the resolution rings overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
        """
        detector_distance = metadata.get("detector_distance", np.nan)
        if "magic_number" in metadata:
            beam_energy = metadata.get("incident_energy", np.nan)
        else:
            beam_energy = metadata.get("beam_energy", np.nan)
        beam_center_x = metadata.get("beam_center_x", np.nan)
        beam_center_y = metadata.get("beam_center_y", np.nan)

        n_rot90 = self._sv_streamctrl.n_rot90
        im_shape = self._sv_streamctrl.current_image_shape  # image shape after rotation in sv
        if n_rot90 == 1 or n_rot90 == 3:
            # get the original shape for consistency in calculations
            im_shape = im_shape[1], im_shape[0]

        if n_rot90 == 1:  # (x, y) -> (y, -x)
            beam_center_x, beam_center_y = beam_center_y, im_shape[1] - beam_center_x
        elif n_rot90 == 2:  # (x, y) -> (-x, -y)
            beam_center_x, beam_center_y = im_shape[1] - beam_center_x, im_shape[0] - beam_center_y
        elif n_rot90 == 3:  # (x, y) -> (-y, x)
            beam_center_x, beam_center_y = im_shape[0] - beam_center_y, beam_center_x

        self._formatter_source.data.update(
            detector_distance=[detector_distance],
            beam_energy=[beam_energy],
            beam_center_x=[beam_center_x],
            beam_center_y=[beam_center_y],
        )

        if not self.switch.active:
            self._clear()
            return

        if any(np.isnan([detector_distance, beam_energy, beam_center_x, beam_center_y])):
            self._sv_metadata.add_issue("Metadata does not contain all data for resolution rings")
            self._clear()
            return

        array_beam_center_x = beam_center_x * np.ones(len(self.positions))
        array_beam_center_y = beam_center_y * np.ones(len(self.positions))
        # if '6200 / beam_energy > 1', then arcsin returns nan
        theta = np.arcsin(6200 / beam_energy / self.positions)  # 6200 = 1.24 / 2 / 1e-4
        ring_diams = 2 * detector_distance * np.tan(2 * theta) / 75e-6
        # if '2 * theta > pi / 2 <==> diams < 0', then return nan
        ring_diams[ring_diams < 0] = np.nan

        text_x = array_beam_center_x + ring_diams / 2
        text_y = array_beam_center_y
        ring_text = [str(s) + " Å" for s in self.positions]

        self._source.data.update(
            x=array_beam_center_x,
            y=array_beam_center_y,
            w=ring_diams,
            h=ring_diams,
            text_x=text_x,
            text_y=text_y,
            text=ring_text,
        )
