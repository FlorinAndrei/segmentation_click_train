tar -jcvf ../archive_segmentation_click_train.tar.bz2 `cat .gitignore | grep -v ^! | grep -v "^\." | grep -v ^_ | grep -v notify_settings`
