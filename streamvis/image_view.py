import bottleneck as bn
import numpy as np
from bokeh.models import CheckboxGroup, ColumnDataSource, CustomJS, HoverTool, Label, Range1d
from bokeh.plotting import figure
from PIL import Image as PIL_Image

js_move_zoom = """
    var data = source.data;
    data['{start}'] = [cb_obj.start];
    data['{end}'] = [cb_obj.end];
    source.change.emit();
"""


class ImageView:
    def __init__(
        self,
        height=894,
        width=854,
        image_height=100,
        image_width=100,
        x_start=None,
        x_end=None,
        y_start=None,
        y_end=None,
    ):
        """Initialize image view plot.

        Args:
            height (int, optional): Height of plot area in screen pixels. Defaults to 894.
            width (int, optional): Width of plot area in screen pixels. Defaults to 854.
            image_height (int, optional): Image height in pixels. Defaults to 100.
            image_width (int, optional): Image width in pixels. Defaults to 100.
            x_start (int, optional): Initial x-axis start value. If None, then equals to 0.
                Defaults to None.
            x_end (int, optional): Initial x-axis end value. If None, then equals to image_width.
                Defaults to None.
            y_start (int, optional): Initial y-axis start value. If None, then equals to 0.
                Defaults to None.
            y_end (int, optional): Initial y-axis end value. If None, then equals to image_height.
                Defaults to None.
        """
        if x_start is None:
            x_start = 0

        if x_end is None:
            x_end = image_width

        if y_start is None:
            y_start = 0

        if y_end is None:
            y_end = image_height

        self.zoom_views = []

        plot = figure(
            x_axis_location="above",
            y_axis_location="right",
            x_range=Range1d(x_start, x_end, bounds=(0, image_width)),
            y_range=Range1d(y_start, y_end, bounds=(0, image_height)),
            height=height,
            width=width,
            toolbar_location="left",
            tools="pan,wheel_zoom,save,reset",
        )
        self.plot = plot

        plot.toolbar.logo = None
        plot.toolbar.tools[1].maintain_focus = False

        plot.yaxis.major_label_orientation = "vertical"

        # ---- image glyph
        self._image_source = ColumnDataSource(
            dict(
                image=[np.zeros((1, 1), dtype="float32")],
                x=[x_start],
                y=[y_start],
                dw=[x_end - x_start],
                dh=[y_end - y_start],
            )
        )

        image_renderer = plot.image(
            source=self._image_source,
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            name="image_glyph",
        )
        self.image_glyph = image_renderer.glyph

        # This avoids double update of image values on a client, see
        # https://github.com/bokeh/bokeh/issues/7079
        # https://github.com/bokeh/bokeh/issues/7299
        image_renderer.view.source = ColumnDataSource()

        plot.add_tools(HoverTool(tooltips=[("intensity", "@image")], renderers=[image_renderer]))

        # ---- pixel value text glyph
        self._pvalue_source = ColumnDataSource(dict(x=[], y=[], text=[]))
        plot.text(
            source=self._pvalue_source,
            x="x",
            y="y",
            text="text",
            text_align="center",
            text_baseline="middle",
            text_color="white",
        )

        # ---- horizontal and vertical projection line glyphs
        self._hproj_source = ColumnDataSource(dict(x=[], y=[]))
        plot.line(source=self._hproj_source, x="x", y="y", line_color="greenyellow")

        self._vproj_source = ColumnDataSource(dict(x=[], y=[]))
        plot.line(source=self._vproj_source, x="x", y="y", line_color="greenyellow")

        proj_switch = CheckboxGroup(labels=["Inner Projections"], width=145, margin=(0, 5, 0, 5))
        self.proj_switch = proj_switch

        # ---- image view coordinates label
        self._coord_label = Label(
            x=10, y=10, text="", text_color="white", x_units="screen", y_units="screen"
        )
        plot.add_layout(self._coord_label)

    @property
    def displayed_image(self):
        """Return resized image that is currently displayed (readonly)."""
        return self._image_source.data["image"][0]

    # a reason for the additional boundary checks:
    # https://github.com/bokeh/bokeh/issues/8118
    @property
    def x_start(self):
        """Current x-axis image start value (readonly)."""
        return int(np.floor(max(self.plot.x_range.start, self.plot.x_range.bounds[0])))

    @property
    def x_end(self):
        """Current x-axis image end value (readonly)."""
        return int(np.ceil(min(self.plot.x_range.end, self.plot.x_range.bounds[1])))

    @property
    def y_start(self):
        """Current y-axis image start value (readonly)."""
        return int(np.floor(max(self.plot.y_range.start, self.plot.y_range.bounds[0])))

    @property
    def y_end(self):
        """Current y-axis image end value (readonly)."""
        return int(np.ceil(min(self.plot.y_range.end, self.plot.y_range.bounds[1])))

    def add_as_zoom(self, image_view, line_color="red"):
        """Add an ImageView plot as a zoom view.

        Args:
            image_plot (ImageView): Associated streamvis image view instance.
            line_color (str, optional): Zoom border box color. Defaults to 'red'.
        """
        # ---- activate WheelZoomTool
        image_view.plot.toolbar.active_scroll = image_view.plot.tools[1]

        # ---- add quad glyph of zoom area to the main plot
        area_source = ColumnDataSource(
            dict(
                left=[image_view.x_start],
                right=[image_view.x_end],
                bottom=[image_view.y_start],
                top=[image_view.y_end],
            )
        )

        self.plot.quad(
            source=area_source,
            left="left",
            right="right",
            bottom="bottom",
            top="top",
            line_color=line_color,
            line_width=2,
            fill_alpha=0,
        )

        x_range_cb = CustomJS(
            args=dict(source=area_source), code=js_move_zoom.format(start="left", end="right")
        )
        y_range_cb = CustomJS(
            args=dict(source=area_source), code=js_move_zoom.format(start="bottom", end="top")
        )

        image_view.plot.x_range.js_on_change("start", x_range_cb)
        image_view.plot.x_range.js_on_change("end", x_range_cb)
        image_view.plot.y_range.js_on_change("start", y_range_cb)
        image_view.plot.y_range.js_on_change("end", y_range_cb)

        self.zoom_views.append(image_view)

    def update(self, image, pil_image=None):
        """Trigger an update for the image view plot.

        Args:
            image (ndarray): A source image for image view.
            pil_image (Image, optional): A source image for image view converted to PIL Image.
                Defaults to None.
        """
        if pil_image is None:
            # this makes an extra copy, see https://github.com/python-pillow/Pillow/issues/3336
            pil_image = PIL_Image.fromarray(image.astype(np.float32, copy=False))

        if (
            self.plot.y_range.bounds[1] != pil_image.height
            or self.plot.x_range.bounds[1] != pil_image.width
        ):
            self.plot.x_range.start = 0
            self.plot.x_range.reset_start = 0
            self.plot.x_range.end = pil_image.width
            self.plot.x_range.reset_end = pil_image.width
            self.plot.x_range.bounds = (0, pil_image.width)

            self.plot.y_range.start = 0
            self.plot.y_range.reset_start = 0
            self.plot.y_range.end = pil_image.height
            self.plot.y_range.reset_end = pil_image.height
            self.plot.y_range.bounds = (0, pil_image.height)

        if (
            self.plot.inner_width < self.x_end - self.x_start
            or self.plot.inner_height < self.y_end - self.y_start
        ):
            resized_image = np.asarray(
                pil_image.resize(
                    size=(self.plot.inner_width, self.plot.inner_height),
                    box=(self.x_start, self.y_start, self.x_end, self.y_end),
                    resample=PIL_Image.NEAREST,
                )
            )

        else:
            resized_image = image[self.y_start : self.y_end, self.x_start : self.x_end]

        self._image_source.data.update(
            image=[resized_image],
            x=[self.x_start],
            y=[self.y_start],
            dw=[self.x_end - self.x_start],
            dh=[self.y_end - self.y_start],
        )

        # Draw numbers
        canvas_pix_ratio_x = self.plot.inner_width / (self.x_end - self.x_start)
        canvas_pix_ratio_y = self.plot.inner_height / (self.y_end - self.y_start)
        if canvas_pix_ratio_x > 70 and canvas_pix_ratio_y > 50:
            textv = [
                f"{val:.1f}"
                for val in image[self.y_start : self.y_end, self.x_start : self.x_end].flatten()
            ]
            xv, yv = np.meshgrid(
                np.arange(self.x_start, self.x_end), np.arange(self.y_start, self.y_end)
            )
            self._pvalue_source.data.update(x=xv.flatten() + 0.5, y=yv.flatten() + 0.5, text=textv)
        else:
            self._pvalue_source.data.update(x=[], y=[], text=[])

        # Draw projections
        if self.proj_switch.active:
            im_y_len, im_x_len = resized_image.shape

            h_x = np.linspace(self.x_start + 0.5, self.x_end - 0.5, im_x_len)
            v_x = np.linspace(self.y_start + 0.5, self.y_end - 0.5, im_y_len)

            h_y = bn.nanmean(resized_image, axis=0)
            v_y = bn.nanmean(resized_image, axis=1)

            h_y = _normalize(h_y, self.plot.y_range.start, self.plot.y_range.end)
            v_y = _normalize(v_y, self.plot.x_range.start, self.plot.x_range.end)

            self._hproj_source.data.update(x=h_x, y=h_y)
            self._vproj_source.data.update(x=v_y, y=v_x)
        else:
            self._hproj_source.data.update(x=[], y=[])
            self._vproj_source.data.update(x=[], y=[])

        # Update image view coordinates label text
        self._coord_label.text = (
            f"Y:({self.y_start}, {self.y_end}) X:({self.x_start}, {self.x_end})"
        )

        # Process all accociated zoom views
        for zoom_view in self.zoom_views:
            zoom_view.update(image, pil_image)


def _normalize(vec, start, end):
    vec -= bn.nanmin(vec)

    v_max = bn.nanmax(vec)
    if v_max != 0:
        vec /= v_max

    vec *= (end - start) * 0.2
    vec += start + (end - start) * 0.05

    return vec
