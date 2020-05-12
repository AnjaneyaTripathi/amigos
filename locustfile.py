from locust import HttpLocust, TaskSet, task, between

class UserBehavior(TaskSet):

    @task
    def emp_tests(self):
        response = self.client.get("/emp")
        print('/EMP Response Content: ',response.text, flush = True)
    
    @task
    def events_tests(self):
        response1 = self.client.get("/events")
        print('/EVENTS Response Content: ', response1.text, flush = True)


class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    wait_time = between(3,6)
