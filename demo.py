# -*- coding: utf-8 -*-

from fs_tests import train_net, infer_test, select_method

def main():
  method = "Huang"
  setup = select_method(method)
  
  train_net(setup, sid=2, epochs=3)
  infer_test(setup, sid=2, n_styles=1)
  
if __name__ == '__main__':
  main()