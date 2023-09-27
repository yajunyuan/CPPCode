#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>

template <typename T>
class Queue {
protected:
	// Data
	std::queue<T> _queue;
	typename std::queue<T>::size_type _size_max;

	// Thread gubbins
	std::mutex _mutex;
	std::condition_variable _fullQue;
	std::condition_variable _empty;

	// Exit
	// ԭ�Ӳ���
	std::atomic_bool _quit; //{ false };
	std::atomic_bool _finished; // { false };

public:
	Queue(){
		_quit = ATOMIC_VAR_INIT(false);
		_finished = ATOMIC_VAR_INIT(false);
	}

	bool push(T& data)
	{
		std::unique_lock<std::mutex> lock(_mutex);
		while (!_quit && !_finished)
		{
			//if (_queue.size() < _size_max)
			if (_queue.size() >= 0)
			{
				//_queue.push(std::move(data));
				_queue.push(data);
				_empty.notify_all();
				return true;
			}
			else
			{
				// wait��ʱ���Զ��ͷ��������wait���˻��ȡ��
				_fullQue.wait(lock);
			}
		}

		return false;
	}


	bool pop(T& data)
	{
		std::unique_lock<std::mutex> lock(_mutex);
		while (!_quit)
		{
			if (!_queue.empty())
			{
				//data = std::move(_queue.front());
				data = _queue.front();
				_queue.pop();

				_fullQue.notify_all();
				return true;
			}
			else if (_queue.empty() && _finished)
			{
				return false;
			}
			else
			{
				_empty.wait(lock);
			}
		}
		return false;
	}

	// The queue has finished accepting input
	void finished()
	{
		std::cout << "queue input finished" << std::endl;
		_finished = true;
		_empty.notify_all();
	}

	void quit()
	{
		std::cout << "queue quited" << std::endl;
		_quit = true;
		_empty.notify_all();
		_fullQue.notify_all();
	}

	int length()
	{
		return static_cast<int>(_queue.size());
	}
};