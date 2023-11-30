
dataset_folder='results/manager'

generalization_cfg={
        "number":5,
        "ckpt_parent_folder":'results/generalization_test',
        "agenda": [
        {
            'name': '1.train_on_hiking', 
            'reload_model': False, 
            'use_online_ckpt': False,
            'dataset_folder': dataset_folder,
            'test_only': False,
         },
        {
            'name': '2.test_on_snow',
            'reload_model': False,
            'use_online_ckpt': False,
            'dataset_folder': dataset_folder,
            'test_only': True
        },
        {
            'name': '3.train_on_snow', 
            'reload_model': False, 
            'use_online_ckpt': False,
            'dataset_folder': dataset_folder,
            'test_only': False,
         },
        {
            'name': '4.test_on_hiking',
            'reload_model': False,
            'use_online_ckpt': False,
            'dataset_folder': dataset_folder,
            'test_only': True
        }
    ]   
    }
    
memory_cfg={
        "number":5,
        "ckpt_parent_folder":'results/memory_test',
        "agenda":[
        {
            'name': '1.train_on_snow', 
            'reload_model': False, 
            'use_online_ckpt': False,
            'dataset_folder': dataset_folder,
            'test_only': False
         },
        {
            'name': '2.resume_train_on_hiking',
            'reload_model': True,
            'use_online_ckpt': False,
            'dataset_folder': dataset_folder,
            'test_only': False
        },
        {
            'name': '3.retest_on_snow',
            'reload_model': False,
            'use_online_ckpt': False,
            'dataset_folder': dataset_folder,
            'test_only': True
        },
        {
            'name': '4.directly_train_on_hiking',
            'reload_model': False,
            'use_online_ckpt': False,
            'dataset_folder': dataset_folder,
            'test_only': False
        }, 
    ]
    }