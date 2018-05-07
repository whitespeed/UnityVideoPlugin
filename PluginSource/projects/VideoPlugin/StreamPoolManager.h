#pragma once
extern "C" {
#include <libavformat\avformat.h>
}

#include <iostream>   
#include <list>

#include <stdio.h>
using namespace std;

class StreamPoolManager
{
public:
	list<AVIOContext> *FixStreamList;
	list<AVIOContext> *DynamicStreamList;

	int RegisterAVIOContext(AVIOContext *);

};