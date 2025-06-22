class Agent:
    def __init__(self, name, task_function):
        self.name = name
        self.task_function = task_function

    def handle(self, task):
        print(f"[{self.name}] handling task: {task}")
        return self.task_function(task)


# Define specific task functions for agents
def billing_agent(task):
    return f"Billing info processed for: {task}"

def support_agent(task):
    return f"Support ticket created for: {task}"

def router_agent(task):
    if "billing" in task:
        return billing.handle(task)
    else:
        return support.handle(task)


# Create agent instances
billing = Agent("BillingAgent", billing_agent)
support = Agent("SupportAgent", support_agent)
router = Agent("RouterAgent", router_agent)


# Simulated incoming tasks
tasks = ["billing issue for user123", "technical issue for user456"]

for task in tasks:
    result = router.handle(task)
    print("Result:", result)