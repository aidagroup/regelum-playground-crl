cd $ROOT_DIR/.git && \
  sudo chgrp -R $(id -g -n $(whoami)) . &&\
  sudo chmod -R g+rwX . &&\
  sudo find . -type d -exec chmod g+s '{}' + &&\
  cd ..