#pragma once
#ifndef __CLASSFACTORY_
#define __CLASSFACTORY_

#include <iostream>
#include<string>
#include<map>

typedef void* (*create_fun)();

class InputPluginFactory
{
public:
	~InputPluginFactory() {
		fac = NULL;
	};

	void registClass(std::string name, create_fun fun) {
	}

	static InputPluginFactory& getInstance() {
		if (!fac)
		{
			fac = new InputPluginFactory();
		}

		return *fac;
	}
private:
	std::map<std::string, create_fun> my_map;
	static InputPluginFactory *fac;
	InputPluginFactory() {};  //к╫сп
};

#endif