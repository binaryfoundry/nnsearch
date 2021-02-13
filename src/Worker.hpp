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

    std::unique_ptr<std::thread> thread_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable condition_join_;
    std::function<void()> job;

public:
    Worker(std::function<void()>&& job) :
        job(std::move(job)),
        active(true),
        working(false)
    {
        thread_ = std::make_unique<std::thread>([=] {
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
            condition_.notify_one();
            thread_->join();
        }
    }

    void Notify()
    {
        working = true;
        condition_.notify_one();
    }

    void Join()
    {
        if (working)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_join_.wait(lock, [this] {
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
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] {
                return working || !active;
            });

            if (active && working)
            {
                job();
            }

            condition_join_.notify_one();
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
