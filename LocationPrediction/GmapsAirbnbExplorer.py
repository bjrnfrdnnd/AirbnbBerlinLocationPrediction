import numpy as np
from IPython.display import display
import ipywidgets as widgets
import gmaps
# gmapsAPIKey = 'bla'
# gmaps.configure(api_key=gmapsAPIKey)


class AirbnbExplorer(object):
    """
    Jupyter widget for exploring the Airbnb Berlin dataset.

    The user uses the slider to choose the radius of the heatmap.
    """

    def __init__(self, df):
        self._df = df
        self._heatmap = None
        self._fig = None
        self._sliderMaxIntensity = None
        self._sliderRadius = None
        self._initial_maxIntensity = 1
        self._initial_radius = 15
        self._min_radius = 1
        self._max_radius = 30
        self._min_maxIntensity = 0
        self._max_maxIntensity = 100

        self._probabilities = self._df['probability'].values
        self._locations = self._df[['latitude', 'longitude']].values

        my_dict={}
        my_dict['room_type'] = df['room_type'].values[0]
        my_dict['bedrooms'] = df['bedrooms'].values[0]
        my_dict['accommodates'] = df['accommodates'].values[0]
        my_dict['price'] = df['price'].values[0]

        if df.loc[0,'prediction_method'] == 'prediction':
            my_dict['prediction_kind'] = 'Prediction of values based on random forest'
        else:
            n_values_found = sum(df['class'])
            my_dict['prediction_kind'] = '{} Values were found in the listings corresponding to requested features and locations. So lookup was used'.format(n_values_found)


        title_widget = widgets.HTML(
            '<h3>Airbnb Berlin Location Prediction</h3>'
            '<h4>Data from <a href="http://tomslee.net/airbnb-data">Tom Slees collection of Airbnb data</a></h4>'
            '<h5>feature set: room_type: {room_type}; bedrooms: {bedrooms}; accommodates: {accommodates}; price: {price}</h5>'
            '<h6>Method: {prediction_kind}</h6>'.format(**my_dict)
        )

        map_figure = self._render_map(self._initial_maxIntensity, self._initial_radius)
        self._fig = map_figure
        controlsMaxIntensity = self._render_controlsMaxIntensity(self._initial_maxIntensity)
        controlsRadius = self._render_controlsRadius(self._initial_radius)
        self._container = widgets.VBox([title_widget, controlsMaxIntensity, controlsRadius, map_figure])

    def render(self):
        display(self._container)

    def _on_maxIntensity_change(self, change):
        maxIntensity = self._sliderMaxIntensity.value
        self._heatmap.max_intensity = maxIntensity
        self._total_boxMaxIntensity.value = self._total_maxIntensity_text(maxIntensity)
        return self._container

    def _on_radius_change(self, change):
        radius = self._sliderRadius.value
        self._heatmap.point_radius = radius
        self._total_boxRadius.value = self._total_radius_text(radius)
        return self._container

    def _render_map(self, initial_maxIntensity, initial_radius):
        # fig = gmaps.figure(map_type='HYBRID')
        # fig = gmaps.figure(map_type='ROADMAP')
        fig = gmaps.figure()
        locations = self._locations
        weights = self._probabilities
        self._heatmap = gmaps.heatmap_layer(
            locations,
            weights=weights,
            max_intensity=initial_maxIntensity,
            opacity=0.8,
            point_radius=initial_radius,
            dissipating=True,
        )
        fig.add_layer(self._heatmap)
        l_weights = len(weights)
        N1 = int(l_weights / 100)
        #         N1 = min(N1,20)
        #         N1 = int(l_weights/2)
        #         N1 = l_weights
        N1 = 1000
        N1 = min(N1, len(weights))

        indicesForMarkers = np.argsort(weights)[-N1:]

        weightsMarked = weights[indicesForMarkers]
        locationsMarked = locations[indicesForMarkers]

        textForMarked = ['proba: {}\nlat: {}\nlon: {}'.format(a,b[0],b[1]) for a,b in zip(weightsMarked, locationsMarked)]

        self._symbolLayer = gmaps.symbol_layer(locationsMarked,
                                               hover_text=textForMarked,
                                               info_box_content=textForMarked,
                                               # stroke_opacity=1, # does not do anything
                                               # fill_opacity=1, # does not do anything
                                               stroke_color="rgba(255, 0, 0, 0.0)",
                                               # workaround for not working opacity
                                               fill_color="rgba(255, 0, 0, 0.0)",  # workaround for not working opacity
                                               scale=3)
        fig.add_layer(self._symbolLayer)

        # self._markerLayer = gmaps.marker_layer(locationsMarked,
        #                                        hover_text=textForMarked,
        #                                        info_box_content=textForMarked,
        #                                        )
        # fig.add_layer(self._markerLayer)

        return fig

    def _render_controlsMaxIntensity(self, initial_maxIntensity):
        self._sliderMaxIntensity = widgets.FloatSlider(
            value=initial_maxIntensity,
            min=self._min_maxIntensity,
            max=self._max_maxIntensity,
            description='maximum intensity',
            step=1.e-2,
            continuous_update=False
        )
        self._total_boxMaxIntensity = widgets.Label(value=self._total_maxIntensity_text(initial_maxIntensity))
        self._sliderMaxIntensity.observe(self._on_maxIntensity_change, names='value')
        controls = widgets.HBox(
            [self._sliderMaxIntensity, self._total_boxMaxIntensity],
            layout={'justify_content': 'space-between'}
        )
        return controls

    def _render_controlsRadius(self, initial_radius):
        self._sliderRadius = widgets.IntSlider(
            value=initial_radius,
            min=self._min_radius,
            max=self._max_radius,
            description='Radius',
            continuous_update=False
        )
        self._total_boxRadius = widgets.Label(
            value=self._total_radius_text(initial_radius)
        )
        self._sliderRadius.observe(self._on_radius_change, names='value')
        controls = widgets.HBox(
            [self._sliderRadius, self._total_boxRadius],
            layout={'justify_content': 'space-between'}
        )
        return controls



    def _total_radius_text(self, radius):
        return 'radius: {}'.format(radius)

    def _total_maxIntensity_text(self, maxIntensity):
        return 'maxIntensity: {}'.format(maxIntensity)
