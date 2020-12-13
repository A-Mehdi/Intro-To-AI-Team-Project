Intro-To-AI-TeamProject
=========================

How to Test
---------------------------------------------------------------------------
1.  Put input images into test_images folder

2.  Use Test.py for test
    Or use train_notebook.ipynb and run test cell for test

3.  Test result images will be saved in save_images folder.
---------------------------------------------------------------------------


Models
---------------------------------------------------------------------------------------------
*  Our final trained model is saved in train_model folder named "face2cartoon_params_latest.pt".  
*  Other models are saved for comparison.
---------------------------------------------------------------------------------------------

Results
-------------------------------------------------------------------------------------------
1.  FID score: 140.7905003784768 for 30000 epoch from scratch without additional male pictures.  
2.  FID score: 90.23898659001463 pretrained+60000 epoch with additional male pictures.   
3.  FID score: 78.01624949505901 pretrained model without any modification.  
-------------------------------------------------------------------------------------------
