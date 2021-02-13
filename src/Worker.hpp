#pragma once

#include <algorithm>
#include <memory>
#include <queue>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>

class Worker
{
private:
    std::atomic<bool> active;
    std::atomic<bool> working;

    std::unique_ptr<std::thread> thread;
    std::mutex mutex;
    std::condition_variable condition;
    std::condition_variable condition_join;
    std::function<void()> job;

public:
    Worker(std::function<void()>&& job) :
        job(std::move(job)),
        active(true),
        working(false)
    {
        thread = std::make_unique<std::thread>([=] {
            thread_loop();
        });
    }

    ~Worker()
    {
        Terminate();
    }

    void Terminate()
    {
        if (active)
        {
            active = false;
            condition.notify_one();
            thread->join();
        }
    }

    void Notify()
    {
        working = true;
        condition.notify_one();
    }

    void Join()
    {
        if (working)
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition_join.wait(lock, [this] {
                return !working || !active;
            });
            working = false;
        }
    }

private:
    void thread_loop()
    {
        do
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [this] {
                return working || !active;
            });

            if (active && working)
            {
                job();
            }

            condition_join.notify_one();
            working = false;
        }
        while (active);
    }
};

class WorkerPool
{
private:
    std::vector<std::unique_ptr<Worker>> workers_;

public:
    WorkerPool()
    {
    }

    ~WorkerPool()
    {
        Terminate();
    }

    void AddWorker(std::unique_ptr<Worker> worker)
    {
        workers_.push_back(std::move(worker));
    }

    void Notify()
    {
        for (auto& w : workers_)
        {
            w->Notify();
        }
    }

    void Join()
    {
        for (auto& w : workers_)
        {
            w->Join();
        }
    }

    void Resolve()
    {
        Notify();
        Join();
    }

    void Terminate()
    {
        for (auto& w : workers_)
        {
            w->Terminate();
        }
    }
};
