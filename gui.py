from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scatter import Scatter
from kivy.uix.stencilview import StencilView
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

import rasterio
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from multiprocessing import Process, Queue
import render_sideviews
import os


def run_sideviews_in_process(q, *args, **kwargs):
    multiviews = render_sideviews.sideviews(*args, **kwargs)
    print("finished getting multiviews")
    q.put(multiviews)
    print("finished queueing multiviews")


class Main(App):
    def __init__(self,project_directory='./',labels=['Category 1', 'Category 2', 'Category 3'], colors=[(152,126,181),(255,230,114),(104,146,182),(247,236,225)], **kwargs):
        super().__init__(**kwargs)
        # customizable inputs
        self.project_directory = project_directory
        self.colors = colors
        self.labels = labels
        # initialize other variables
        self.all_bboxes = {} # img_bbox (x0,y0,x1,y1) : (geo_bbox (long0,lat0,long1,lat1), img_contour (contour), geo_contour (Polygon), label (str))
        self.current_bbox = {'img_bbox': None, 'geo_bbox': None, 'img_contour': None, 'geo_contour': None}
        self.labeling = False
        self.label_color = {}
        for label,color in zip(self.labels,self.colors):
            self.label_color[label]=tuple(val/255 for val in color)
    
    def build(self):
        layout = BoxLayout(orientation='horizontal')
        # left ortho
        self.left_box = StencilBoxLayout(orientation='vertical',size_hint=(0.6,1))
        self.left_orthomosaic = Ortho(self,size_hint=(1,0.95))
        self.left_box.add_widget(self.left_orthomosaic)
        self.main_tools = MainTools(self,size_hint=(1,0.05))
        self.left_box.add_widget(self.main_tools)
        layout.add_widget(self.left_box)
        # right supplementary
        self.right_supplementary = BoxLayout(orientation='vertical',size_hint=(0.4,1))
        self.labeling_tools = LabelingTools(self,size_hint=(1,.1))
        self.side_views = SideViews(size_hint=(1,.9))
        self.right_supplementary.add_widget(self.labeling_tools)
        self.right_supplementary.add_widget(self.side_views)
        layout.add_widget(self.right_supplementary)
        # return
        return layout

class StencilBoxLayout(StencilView, BoxLayout):
    """
    layout container for orthomosaic on the left
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

class Ortho(Scatter):
    """
    orthomosaic object
    """
    def __init__(self, main_app, **kwargs):
        super().__init__(**kwargs)
        self.main_app = main_app
        self.drawing=False
        self.x0, self.y0 = -1, -1

        # load up saved labels
        if os.path.exists(self.main_app.project_directory+'labels.gpkg'):
            gdf = gpd.read_file(self.main_app.project_directory+'labels.gpkg')
            print(gdf.head()) ##
            for i,row in gdf.iterrows():
                img_coord = row['img_coords'].split(',')
                img_coord = tuple(int(coord) for coord in img_coord)
                geo_coord = row['geo_coords'].split(',')
                geo_coord = tuple(float(coord) for coord in geo_coord)
                img_contour = []
                for coords in row['img_contours'].split(','):
                    x,y = coords.split()
                    img_contour.append([int(x),int(y)])
                img_contour = np.array(img_contour, np.int32)
                geo_contour = row['contour_polygons']
                self.main_app.all_bboxes[img_coord] = (geo_coord,img_contour,geo_contour,row['species_observed'])

        # load up orthos
        self.ortho_raster = rasterio.open(main_app.project_directory+'exports/orthomosaic.tif')
        self.raw_ortho = self.ortho_raster.read().transpose(1, 2, 0)[:, :, :3][:, :, [2, 1, 0]]
        self.labeled_ortho = self.raw_ortho.copy()
        for (x0,y0,x1,y1),(geo_coords,img_contour,geo_contour,label) in self.main_app.all_bboxes.items():
            color = [_*255 for _ in self.main_app.label_color[label][2::-1]]
            cv2.rectangle(self.labeled_ortho, (x0,y0), (x1,y1), color, 20)
            cv2.drawContours(self.labeled_ortho, [img_contour], 0, color, 3)
        self.working_ortho = self.labeled_ortho.copy()

        # Convert numpy array to texture
        self.texture = Texture.create(size=(self.working_ortho.shape[1], self.working_ortho.shape[0]), colorfmt='bgr')
        self.texture.blit_buffer(self.working_ortho.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        self.aspect_ratio = self.texture.width / self.texture.height
        
        with self.canvas:
            self.rect = Rectangle(texture=self.texture, pos=(0, 0), size=self.texture.size)
        self.bind(size=self._update_rect, pos=self._update_rect)
    
    def reload_trees(self,region=None):
        self.labeled_ortho = self.raw_ortho.copy()
        for (x0,y0,x1,y1),(geo_coords,img_contour,geo_contour,label) in self.main_app.all_bboxes.items():
            cv2.rectangle(self.labeled_ortho, (x0,y0), (x1,y1), [_*255 for _ in self.main_app.label_color[label][2::-1]], 20)
            # TODO - add contours
        self.working_ortho = self.labeled_ortho.copy()
        self.update(region)
    
    def _update_rect(self, instance, value):
        if self.width / self.height > self.aspect_ratio:
            new_width = self.height * self.aspect_ratio
            new_height = self.height
        else:
            new_width = self.width
            new_height = self.width / self.aspect_ratio
        self.rect.size = (new_width, new_height)
        self.rect.pos = ((self.width - new_width) / 2, (self.height - new_height) / 2)
    
    def to_image_coords(self, touch_x, touch_y):
        local_x, local_y = self.to_local(touch_x, touch_y)
        ratio = min(self.width / self.texture.width, self.height / self.texture.height)
        x = int((local_x - (self.width - self.texture.width * ratio) / 2) / ratio)
        y = int(((local_y -(self.height - self.texture.height * ratio) / 2) / ratio))
        return (max(0, min(x, self.texture.width - 1)), max(0, min(y, self.texture.height - 1)))

    def on_touch_down(self,touch):
        if self.main_app.left_box.collide_point(*touch.pos):
            if touch.is_mouse_scrolling: # zoom
                if touch.button == 'scrolldown':  # zoom in
                    self.scale = min(10, self.scale * 1.1)
                elif touch.button == 'scrollup':  # zoom out
                    self.scale = max(1, self.scale * 0.8)
            elif not self.drawing and not self.main_app.labeling and touch.button == 'left':  # draw
                self.drawing = True
                self.x0, self.y0 = self.to_image_coords(touch.x, touch.y)
            elif touch.button == 'right':  # pan
                self.do_translation = True
            return True
        return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            if self.drawing: # draw
                self.working_ortho = self.labeled_ortho.copy()
                cv2.rectangle(self.working_ortho, (self.x0, self.y0), self.to_image_coords(touch.x, touch.y), (0, 0, 255), 20)
                self.update((self.x0,self.y0))
            elif self.do_translation: # pan
                self.x += touch.dx
                self.y += touch.dy
            return True
        return super().on_touch_move(touch)
    
    def on_touch_up(self,touch):
        x,y = self.to_image_coords(touch.x, touch.y)
        if self.collide_point(*touch.pos):
            if self.x0==x and self.y0==y:
                self.drawing = False
                for img_coords,(geo_coords,img_contour,geo_contour,_) in self.main_app.all_bboxes.items():
                    x0,y0,x1,y1 = img_coords
                    if (x0<=x<=x1 and (-10<y-y0<10 or -10<y-y1<10)) or (y0<=y<=y1 and (-10<x-x0<10 or -10<x-x1<10)): # previous bbox
                        self.main_app.labeling = True
                        self.main_app.labeling_tools.update_btns()
                        self.working_ortho = self.labeled_ortho.copy()
                        cv2.rectangle(self.working_ortho, (x0,y0), (x1,y1), (0, 0, 255), 20)
                        cv2.drawContours(self.working_ortho, [img_contour], 0, (0,0,255), 3)
                        self.update()
                        self.main_app.current_bbox = {'img_bbox':img_coords, 'geo_bbox':geo_coords, 'img_contour':img_contour, 'geo_contour':geo_contour}
                        self.main_app.main_tools.update_instructions("update label")
                        return True
            elif self.drawing: # end drawing bbox
                self.drawing = False
                self.main_app.side_views.loading_views()
                self.main_app.main_tools.update_instructions("loading side views")
                x0,x1,y0,y1 = min(self.x0,x),max(self.x0,x),min(self.y0,y),max(self.y0,y)
                self.main_app.current_bbox['img_bbox'] = (x0,y0,x1,y1)
                self.main_app.current_bbox['geo_bbox'] = (self.ortho_raster.transform*(x0,y0)+self.ortho_raster.transform*(x1,y1))
                ## for debugging
                print(f"img_bbox (x0,y0,x1,y1) is {self.main_app.current_bbox['img_bbox']}")
                print(f"geo_bbox is {self.main_app.current_bbox['geo_bbox']}")
                # find roi
                img_contour,mask_roi = render_sideviews.ortho_mask(self.raw_ortho[y0-20:y1+20,x0-20:x1+20,:], self.main_app.current_bbox['geo_bbox'], (x0,y0), self.ortho_raster.transform)
                img_contour = np.array([[x+x0-20,y+y0-20] for [[x,y]] in img_contour])
                self.main_app.current_bbox['img_contour'] = img_contour
                self.main_app.current_bbox['geo_contour'] = mask_roi
                cv2.drawContours(self.working_ortho, [img_contour], 0, (0,0,255), 3)
                self.update((x0,y0))
                # render sideviews
                q = Queue()
                p = Process(target=run_sideviews_in_process, args=(q, mask_roi))
                p.start()
                current_multiviews = q.get()
                p.kill()
                # show sideviews
                self.main_app.side_views.show_views(current_multiviews)
                # solicit labeling input
                self.main_app.labeling = True
                self.main_app.labeling_tools.update_btns()
                self.main_app.main_tools.update_instructions("select a label")     
            elif self.do_translation: # end panning
                self.do_translation = False
            return True

        return super().on_touch_up(touch)

    def update(self,region=None,d=1000):
        if region:
            x0,y0 = region
            update_region = self.working_ortho[y0-d:y0+d, x0-d:x0+d].tobytes()
            self.texture.blit_buffer(update_region, colorfmt='bgr', bufferfmt='ubyte', pos=(x0-d, y0-d), size=(2*d,2*d))
        else:
            self.texture.blit_buffer(self.working_ortho.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        self.canvas.ask_update()

class MainTools(BoxLayout):
    def __init__(self, main_app, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.main_app = main_app
        self.padding = 5
        self.spacing = 2
        # export
        export_btn = Button(text='save',size_hint=(0.15,1))
        export_btn.bind(on_press=self.export_btn)
        self.add_widget(export_btn)
        # instructions text
        self.instructions = Label(text="Draw a bounding box by left clicking then dragging",size_hint=(0.85,1))
        with self.instructions.canvas.before:
            Color(0, 0, 0, 0.5)
            self.rect = Rectangle(pos=self.instructions.pos, size=self.instructions.size)
        self.instructions.bind(size=self.update_rect, pos=self.update_rect)
        self.add_widget(self.instructions)
    
    def update_instructions(self,instructions):
        self.remove_widget(self.instructions)
        self.instructions = Label(text=instructions,size_hint=(0.85,1))
        with self.instructions.canvas.before:
            Color(0, 0, 0, 0.5)
            self.rect = Rectangle(pos=self.instructions.pos, size=self.instructions.size)
        self.instructions.bind(size=self.update_rect, pos=self.update_rect)
        self.add_widget(self.instructions)

    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
    
    def export_btn(self,event):
        all_img_coords,all_geo_coords,ids,labels,img_contours,bbox_polygons,contour_polygons = [],[],[],[],[],[],[]
        for img_coords,(geo_coords,img_contour,geo_contour,label) in self.main_app.all_bboxes.items():
            all_img_coords.append(str(img_coords)[1:-1])
            x0,y0,x1,y1 = img_coords
            all_geo_coords.append(str(geo_coords)[1:-1])
            long0,lat0,long1,lat1 = geo_coords
            ids.append(0) ## #
            labels.append(label)
            img_contours.append(str(img_contour).replace("]\n [", ",")[2:-2])
            bbox_polygons.append(Polygon(((long0,lat0),(long1,lat0),(long1,lat1),(long0,lat1))))
            contour_polygons.append(geo_contour) 
        labels = gpd.GeoDataFrame({
            'observed_tree_id': ids,
            'species_observed': labels,
            'img_coords': all_img_coords,
            'geo_coords': all_geo_coords,
            'img_contours': img_contours,
            'bbox_polygons': bbox_polygons,
            'contour_polygons': contour_polygons
        }, crs='EPSG:4326', geometry='bbox_polygons')
        labels.to_file(self.main_app.project_directory+'labels.gpkg', driver='GPKG')
        self.main_app.main_tools.update_instructions('saved!')
        print(labels.head())
        
class LabelingTools(BoxLayout):
    def __init__(self, main_app, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.main_app = main_app
        self.padding = 5
        self.spacing = 2
        self.buttons = []
        # cancel button
        btn = Button(text='cancel',disabled=not self.main_app.labeling,font_size=12)
        btn.bind(on_press=self.cancel_btn)
        self.buttons.append(btn)
        # delete button
        btn = Button(text='delete',disabled=not self.main_app.labeling,font_size=12)
        btn.bind(on_press=self.delete_btn)
        self.buttons.append(btn)
        # labeling buttons
        for label,color in self.main_app.label_color.items():
            btn = Button(text=label,background_color=color,background_normal='',disabled=not self.main_app.labeling,font_size=12)
            btn.bind(on_press=self.label_btn)
            self.buttons.append(btn)
        for button in self.buttons:
            self.add_widget(button)
    
    def update_btns(self):
        for btn in self.buttons:
            btn.disabled = not self.main_app.labeling

    def cancel_btn(self,event):
        if self.main_app.labeling:
            self.main_app.labeling = False
            self.update_btns()
            # reset
            self.main_app.left_orthomosaic.working_ortho = self.main_app.left_orthomosaic.labeled_ortho.copy()
            self.main_app.left_orthomosaic.update(self.main_app.current_bbox['img_bbox'][0:2])
            self.main_app.current_bbox = {'img_bbox': None, 'geo_bbox': None, 'img_contour': None, 'geo_contour': None}
            self.main_app.side_views.reset_views()

    def label_btn(self,event):
        if self.main_app.labeling:
            self.main_app.labeling = False
            self.update_btns()
            # classify
            self.main_app.all_bboxes[self.main_app.current_bbox['img_bbox']] = (self.main_app.current_bbox['geo_bbox'],self.main_app.current_bbox['img_contour'],self.main_app.current_bbox['geo_contour'],event.text)
            self.main_app.main_tools.update_instructions(f'Added box as {event.text}')
            # update
            x0,y0 = self.main_app.current_bbox['img_bbox'][0:2]
            color = [_*255 for _ in event.background_color[2::-1]]
            cv2.rectangle(self.main_app.left_orthomosaic.labeled_ortho, (self.main_app.current_bbox['img_bbox'][0:2]), self.main_app.current_bbox['img_bbox'][2:], color, 20)
            cv2.drawContours(self.main_app.left_orthomosaic.labeled_ortho, [self.main_app.current_bbox['img_contour']], 0, color, 3)
            self.main_app.left_orthomosaic.working_ortho = self.main_app.left_orthomosaic.labeled_ortho.copy()
            self.main_app.left_orthomosaic.update(self.main_app.current_bbox['img_bbox'][0:2])
            self.main_app.current_bbox = {'img_bbox': None, 'geo_bbox': None, 'img_contour': None, 'geo_contour': None}
            self.main_app.side_views.reset_views()

    def delete_btn(self,event):
        if self.main_app.labeling:
            self.main_app.labeling = False
            self.update_btns()
            # delete
            self.main_app.main_tools.update_instructions('deleted bounding box')
            self.main_app.all_bboxes.pop(self.main_app.current_bbox['img_bbox'], None)
            # update
            self.main_app.left_orthomosaic.reload_trees(self.main_app.current_bbox['img_bbox'][0:2])
            self.main_app.current_bbox = {'img_bbox': None, 'geo_bbox': None, 'img_contour': None, 'geo_contour': None}

class SideViews(StackLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'lr-tb'
        self.padding = 5
        self.spacing = 5
        with self.canvas.before:
            Color(0, 0, 0, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(size=self._update_rect, pos=self._update_rect)
        # default label
        self.default_label = Label(text='Draw a bounding box to see side views')
        self.add_widget(self.default_label)
        # loading label
        self.loading_label = Label(text='Loading side views')
    
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
    
    def reset_views(self):
        self.clear_widgets()
        self.add_widget(self.default_label)

    def loading_views(self):
        self.clear_widgets()
        self.add_widget(self.loading_label)
        print("loading")
        self.canvas.ask_update() ## broken
    
    def show_views(self,current_multiviews):
        self.clear_widgets()
        self.multiviews = []
        if not current_multiviews:
            self.add_widget(Label(text='No views to display'))
            return True
        for view in current_multiviews[:10]:
            view = view.astype(np.uint8)
            texture = Texture.create(size=(view.shape[1], view.shape[0]), colorfmt='bgr')
            texture.blit_buffer(view.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            img = Image(texture=texture,size_hint=(0.5, None))
            img.allow_stretch = True # deprecated?
            img.keep_ratio = True
            self.add_widget(img)
            self.multiviews.append(img)
        for view in self.multiviews:
            view.canvas.ask_update()
        self.canvas.ask_update()

        
if __name__ == '__main__':
    Main(project_directory='./example_hawaii/').run()