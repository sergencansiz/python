class DatasetObject(object):
    
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        
    def add_image_info(self , image_array , image_path , image_id , object_id , class_name , class_id):
        
        image_info_ = {'object_id': image_id,
                       'object_class': class_name,
                       'image_path': image_path,
                       'image_array': image_array,
                       'class_id' : class_id}
        self.image_info.append(image_info_)
    
    def parse_xml(filename):
        boxes = list()
        names = list()
        root = ET.parse(filename).getroot() 
        for box in root.findall('.//object'):
            xmin = int(box.find('.//bndbox/xmin').text)
            xmax = int(box.find('.//bndbox/xmax').text)
            ymin = int(box.find('.//bndbox/ymin').text)
            ymax = int(box.find('.//bndbox/ymax').text)
            name = root.find('.//name').text
            coords = [xmin , ymin , xmax , ymax]
            names.append(name)
            boxes.append(coords)
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        path = root.find('.//path').text
        filename = root.find('filename').text
        filename = filename[:-4]
        return boxes, width, height , names , path , filename
    
    def get_image_arrays(self , annot_path , class_object):
        
        count = 0
        for i in listdir('pizza/annots'):
            parsed = parse_xml(annot_path +'/' + i)
            img = cv2.imread(parsed[4])
            for wh in range(0 , len(parsed[0])):
                crop_img = img[parsed[0][wh][1]:parsed[0][wh][3] , parsed[0][wh][0]:parsed[0][wh][2] ]
                
                class_name = parsed[3][wh]
                keys = list(class_object.keys())  
                values = list(class_object.values())
                class_id = keys[values.index(class_name)]
                
                resized_img = cv2.resize(crop_img,(32,32))
                self.add_image_info( image_array = resized_img , image_path = parsed[4],
                                   object_id = count , class_name=parsed[3][wh] , image_id= 0 , class_id=class_id )
                print('İmage : ' + parsed[4] + ' added to dataset with class name "' + parsed[3][wh]  +'".'  )
                count = count + 1
    
    def prepare_dataset(self , path , class_object , test_size):
        
        self.get_image_arrays(path , class_object)
        
        img_arrays = np.array([d['image_array'] for d in self.image_info if 'image_array' in d])
        class_name = np.array([d['object_class'] for d in self.image_info if 'object_class' in d])
        class_ids  = np.array([d['class_id'] for d in self.image_info if 'class_id' in d])
        
        all_idx = range(0 , len(img_arrays))
        test_idx = np.random.choice(np.arange(len(img_arrays)), test_size, replace=False)
        train_idx = np.delete(all_idx , test_idx) 
        
        
        self.testX = img_arrays[test_idx]
        self.testY = class_ids[test_idx]
        
        self.trainX = img_arrays[train_idx]        
        self.trainY = class_ids[train_idx]

        #trainX = img_arrays
        #trainY = class_ids
        
        return print('Image dataset succesfully saved.')
