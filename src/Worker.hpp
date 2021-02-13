#pragma once


#include <memory>
#include <queue>
#include <atomic>
#include <thread>

#include <functional>

#include <algorithm>
#include <mutex>
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
 
    void Loop();

public:
    Worker(std::function<void()>&& job);
    virtual ~Worker();

    void Terminate();
    void Notify();

    void Join();
};

class WorkerPool
{
private:
    std::vector<std::unique_ptr<Worker>> workers_;

    void Notify();
    void Join();
    void Terminate();

public:
    WorkerPool();
    virtual ~WorkerPool();

    void AddWorker(std::unique_ptr<Worker> worker);
    void Resolve();
};
