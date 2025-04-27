### Questions to Ask
* Is the SQLDataset class sufficiently efficient?
* Is involution on the deeper layers (the ones with more channels and less spatial dims) a good idea? - yes


### Making the Train/Dev/Test Split
* Since we have 18,032 images, we can have a split of 90/5/5, which gives roughly 900 images each in the val and test
* So, this means that every 2 out of 20 (1 for val and 1 for test) will go to val and test 
* I can use 
    SELECT * FROM table_name WHERE MOD(idx, 20) < 18 -- 90% for training
    SELECT * FROM table_name WHERE MOD(idx, 20) = 0 -- 5% for val
    SELECT * FROM table_name WHERE MOD(idx, 20) > 18 -- 5% for test 

